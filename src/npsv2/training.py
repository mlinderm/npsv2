import itertools, logging, random
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from .images import load_example_dataset, _extract_metadata_from_first_example


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
            layers.MaxPooling2D((2, 2)),  # Not in original model, but included here to reduce size
            layers.Flatten(),
            layers.Dense(
                4096,
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
    encoder = tf.keras.models.Sequential(
        [
            layers.Input(input_shape),
            layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
            layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation=tf.nn.relu),
        ]
    )
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
        itertools.combinations(range(3), 2), itertools.product(range(replicates), repeat=2),
    )
    mismatch_pairs = [((ac1, r1), (ac2, r2)) for ((ac1, ac2), (r1, r2)) in mismatch_pairs]

    class_num_pairs = min(len(match_pairs), len(mismatch_pairs), pairs_per_variant // 2)

    pairs = random.sample(match_pairs, class_num_pairs) + random.sample(mismatch_pairs, class_num_pairs)
    return zip(*pairs), ([1] * class_num_pairs + [0] * class_num_pairs)


def _variant_to_training_pairs(features, original_label):
    replicates = features["sim/0/images"].shape[0]

    images = tf.stack([features["sim/0/images"], features["sim/1/images"], features["sim/2/images"]])
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)

    (query_indices, support_indices), pair_labels = _random_pair_indices(replicates, 8)

    image_tensors = {
        "query": tf.gather_nd(images, query_indices),
        "support": tf.gather_nd(images, support_indices),
    }
    label_tensor = tf.constant(pair_labels, dtype=tf.uint64)

    # Construct data to permit "flat_map" and thus more flexible batching downstream
    return tf.data.Dataset.from_tensor_slices((image_tensors, label_tensor))


def _variant_to_test_pairs(features, original_label):
    query_image = tf.image.convert_image_dtype(features["image"], dtype=tf.float32)
    query_images = tf.stack([query_image, query_image, query_image])
    support_images = tf.image.convert_image_dtype(
        tf.stack([features["sim/0/images"][0], features["sim/1/images"][0], features["sim/2/images"][0],]),
        dtype=tf.float32,
    )
    image_tensors = {
        "query": query_images,
        "support": support_images,
    }
    return image_tensors, original_label


def distance_accuracy(y_true, y_pred, threshold=0.5):
    """Compute classification accuracy with a fixed threshold on distances."""
    return tf.keras.backend.mean(tf.math.equal(y_true, tf.cast(y_pred < threshold, y_true.dtype)), axis=-1)


def train(tfrecords_path: str, model_path: str):
    image_shape, replicates = _extract_metadata_from_first_example(tfrecords_path)

    model = _siamese_convsim_model(image_shape)
    model.summary()
    optimizer = tf.keras.optimizers.Adam(lr=0.004)
    model.compile(
        optimizer=optimizer,
        loss=tfa.losses.contrastive_loss,
        metrics=[distance_accuracy],
    )

    # model = _siamese_networks_model(image_shape)

    # # TODO: Decay learning rate, e.g.
    # # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    # #     initial_learning_rate=params.learning_rate,
    # #     decay_steps=10000, # e.g. set to the number of steps per epoch
    # #     decay_rate=0.99)
    # optimizer = tf.keras.optimizers.Adam(lr=0.004)
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
    train_dataset = example_dataset.flat_map(_variant_to_training_pairs).batch(16)

    # TODO: Add early stopping callback
    # TODO: Reserve some training data for validation
    history = model.fit(train_dataset, epochs=5)

    # TODO: Plot training and validation curves

    logging.info("Saving model in %s", model_path)
    model.save(model_path)

    genotype_concordance, nonreference_concordance, _ = evaluate_model(model, example_dataset)
    logging.info(
        "Accuracy - Genotype concordance: %f, Non-reference Concordance: %f",
        genotype_concordance,
        nonreference_concordance,
    )


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

    # Unzip test dataset
    y_pred = []
    y_true = []
    for images, label in test_dataset:
        y_pred.append(tf.reshape(model.predict(images), (-1, 3)))
        y_true.append(label)
    y_pred = tf.concat(y_pred, axis=0)
    y_true = tf.concat(y_true, axis=0)

    y_pred = tf.nn.softmax(-y_pred, axis=1)  # For models that produce distances

    # Use keras mean to avoid type inference
    genotype_concordance = tf.keras.backend.mean(tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred))

    # Non-reference concordance collapses heterozygous and homozygous alternate
    y_pred_label = tf.math.argmax(y_pred, axis=1)
    y_pred_present = tf.math.greater(y_pred_label, 0)
    y_true_present = tf.math.greater(y_true, 0)

    nonreference_concordance = tf.keras.backend.mean(tf.math.equal(y_pred_present, y_true_present))

    # Generate the confusion matrix
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred_label, num_classes=3,)

    return genotype_concordance, nonreference_concordance, confusion_matrix
