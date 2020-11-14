import datetime, functools, itertools, logging, os, random
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from .images import load_example_dataset, _extract_metadata_from_first_example, _features_variant
from . import npsv2_pb2

# Adapted from: https://github.com/few-shot-learning/Keras-FewShotLearning/blob/master/keras_fsl/models/encoders/koch_net.py
def _siamese_networks_model(input_shape):
    kernel_initializer = RandomNormal(0.0, 0.01)
    bias_initializer = RandomNormal(0.5, 0.01)

    encoder = tf.keras.models.Sequential(
        [
            layers.Input(input_shape),
            layers.Conv2D(
                64,
                (10, 10),
                activation=tf.nn.relu,
                kernel_regularizer=l2(),
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(
                128,
                (7, 7),
                activation=tf.nn.relu,
                kernel_regularizer=l2(),
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(
                128,
                (4, 4),
                activation=tf.nn.relu,
                kernel_regularizer=l2(),
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(
                256,
                (4, 4),
                activation=tf.nn.relu,
                kernel_regularizer=l2(),
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            ),
            # layers.MaxPooling2D((2, 2)),  # Not in original model, but included here to reduce size
            layers.Flatten(),
            layers.Dense(
                4096,  # Original size in the paper
                activation=tf.nn.sigmoid,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            ),
        ],
        name="encoder",
    )

    query = layers.Input(input_shape, name="query")
    support = layers.Input(input_shape, name="support")

    embeddings = [encoder(query), encoder(support)]

    output = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))(embeddings)
    output = layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True)(output)
    return tf.keras.Model(inputs=[query, support], outputs=output)


def _euclidean_distance(tensors):
    sum_square = tf.math.reduce_sum(tf.math.squared_difference(tensors[0], tensors[1]), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def _siamese_convsim_model(input_shape):
    # encoder = tf.keras.models.Sequential(
    #     [
    #         layers.Input(input_shape),
    #         layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
    #         layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    #         layers.MaxPooling2D((2, 2)),
    #         layers.Dropout(0.25),
    #         layers.Flatten(),
    #         layers.Dense(128, activation=tf.nn.relu),
    #     ]
    # )

    encoder = tf.keras.models.Sequential([layers.Input(input_shape)])
    for i in range(4):
        encoder.add(layers.Conv2D(64, (3, 3), padding="same"))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.Dropout(0.2))
        encoder.add(layers.Activation(tf.nn.relu))
        encoder.add(layers.MaxPooling2D((2, 2)))
    encoder.add(layers.Flatten())

    encoder.summary()

    query = layers.Input(input_shape, name="query")
    support = layers.Input(input_shape, name="support")

    embeddings = [encoder(query), encoder(support)]

    output = layers.Lambda(_euclidean_distance)(embeddings)
    return tf.keras.Model(inputs=[query, support], outputs=output)


def _random_pair_indices(replicates, pairs_per_variant):
    match_pairs = itertools.product(range(3), itertools.combinations(range(replicates), 2))
    match_pairs = [((ac, r1), (ac, r2)) for (ac, (r1, r2)) in match_pairs]

    mismatch_pairs = itertools.product(
        itertools.combinations(range(3), 2),
        itertools.product(range(replicates), repeat=2),
    )
    mismatch_pairs = [((ac1, r1), (ac2, r2)) for ((ac1, ac2), (r1, r2)) in mismatch_pairs]

    class_num_pairs = min(len(match_pairs), len(mismatch_pairs), pairs_per_variant // 2)

    pairs = random.sample(match_pairs, class_num_pairs) + random.sample(mismatch_pairs, class_num_pairs)
    return zip(*pairs), ([1] * class_num_pairs + [0] * class_num_pairs)


def _variant_to_training_pairs(features, original_label, replicates):
    images = tf.image.convert_image_dtype(features["sim/images"], dtype=tf.float32)

    (query_indices, support_indices), pair_labels = _random_pair_indices(replicates, 8)

    image_tensors = {
        "query": tf.gather_nd(images, query_indices),
        "support": tf.gather_nd(images, support_indices),
    }
    label_tensor = tf.constant(pair_labels, dtype=tf.uint64)

    # Construct data to permit "flat_map" and thus more flexible batching downstream
    return tf.data.Dataset.from_tensor_slices((image_tensors, label_tensor))


def distance_accuracy(y_true, y_pred, threshold=0.5):
    """Compute classification accuracy with a fixed threshold on distances."""
    return tf.keras.backend.mean(tf.math.equal(y_true, tf.cast(y_pred < threshold, y_true.dtype)), axis=-1)


def train(params, tfrecords_path: str, model_path: str):
    image_shape, replicates = _extract_metadata_from_first_example(tfrecords_path)

    model = _siamese_convsim_model(image_shape)
    model.summary()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=params.learning_rate,
        decay_steps=10000,  # e.g. set to the number of steps per epoch
        decay_rate=0.99,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss=tfa.losses.contrastive_loss,
        metrics=[distance_accuracy],
    )

    # model = _siamese_networks_model(image_shape)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.004)
    # model.compile(
    #     optimizer=optimizer,
    #     loss=tf.keras.losses.binary_crossentropy,
    #     metrics=["binary_accuracy"],
    # )

    example_dataset = load_example_dataset(
        tfrecords_path,
        with_label=True,
        with_simulations=True,
    )
    variant_to_training_pairs = functools.partial(_variant_to_training_pairs, replicates=replicates)
    train_dataset = (
        example_dataset.shuffle(1000, reshuffle_each_iteration=True).flat_map(variant_to_training_pairs).batch(16)
    )

    # TODO: Reserve some training data for validation
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)

    checkpoint_filepath = os.path.join(params.tempdir, "checkpoint")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, save_weights_only=True, monitor="loss", mode="min", save_best_only=True
    )

    callbacks=[early_stopping, checkpoint_callback]
    
    if params.log_dir:
        log_dir = os.path.join(params.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        logging.info("Logging TensorBoard data to: %s", log_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)


    model.fit(train_dataset, epochs=params.epochs, callbacks=callbacks)
   
    # Load best model
    model.load_weights(checkpoint_filepath)

    # TODO: Further evaluation or validation

    logging.info("Saving model in: %s", model_path)
    model.save(model_path)


def _variant_to_test_pairs(features, original_label):
    #query_image = tf.image.convert_image_dtype(features["image"], dtype=tf.float32)
    query_images = tf.stack([tf.image.convert_image_dtype(features["image"], dtype=tf.float32)] * 3)
    support_images = tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32)

    # support_images = tf.image.convert_image_dtype(features["sim/images"][:,0]
    #     tf.stack(
    #         [
    #             features["sim/0/images"][0],
    #             features["sim/1/images"][0],
    #             features["sim/2/images"][0],
    #         ]
    #     ),
    #     dtype=tf.float32,
    # )
    image_tensors = {
        "query": query_images,
        "support": support_images,
    }
    return image_tensors, original_label, features["variant/encoded"]


def evaluate_model(model, dataset):
    if isinstance(model, str):
        model = tf.keras.models.load_model(
            model,
            custom_objects={"distance_accuracy": distance_accuracy, "contrastive_loss": tfa.losses.contrastive_loss},
        )
    assert isinstance(model, tf.keras.Model), "Valid model or path not provided"

    if isinstance(dataset, str):
        dataset = load_example_dataset(dataset, with_label=True, with_simulations=True)

    test_dataset = dataset.map(_variant_to_test_pairs)

    # Unzip test dataset into pandas dataframe with genotypes and variant annotations
    rows = []
    for images, label, encoded_variant in test_dataset:
        # Extract metadata for the variant
        variant_proto = npsv2_pb2.StructuralVariant.FromString(encoded_variant.numpy())

        # Predict genotype
        genotype_probabilities = tf.reshape(model.predict(images), (-1, 3))
        genotype_probabilities = tf.nn.softmax(-genotype_probabilities, axis=1)  # For models that produce distances

        if tf.math.argmax(genotype_probabilities, axis=1) != label and label == 0:
            print(variant_proto, label, genotype_probabilities)
            assert False
            

        rows.append(
            pd.DataFrame(
                dict(SVLEN=variant_proto.svlen, LABEL=label, AC=tf.math.argmax(genotype_probabilities, axis=1))
            )
        )

    table = pd.concat(rows, ignore_index=True)
    table["MATCH"] = table.LABEL == table.AC

    genotype_concordance = np.mean(table.MATCH)
    nonreference_concordance = np.mean((table.LABEL > 0) == (table.AC > 0))
    confusion_matrix = pd.crosstab(table.LABEL, table.AC, rownames=["Truth"], colnames=["Test"], margins=True)

    svlen_bins = pd.cut(np.abs(table.SVLEN), [50, 100, 300, 1000, np.iinfo(np.int32).max], right=False)
    print(table.groupby(svlen_bins)["MATCH"].mean())

    return genotype_concordance, nonreference_concordance, confusion_matrix
