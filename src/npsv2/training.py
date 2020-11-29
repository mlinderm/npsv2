import datetime, functools, itertools, logging, os, random, sys
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from scipy.spatial import distance
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from .images import load_example_dataset, _extract_metadata_from_first_example, _features_variant
from . import npsv2_pb2




def _encoder_model(input_shape):
    encoder = tf.keras.models.Sequential([layers.Input(input_shape)], name="encoder")
    for i in range(4):
        encoder.add(layers.Conv2D(64, (3, 3), padding="same"))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.Dropout(0.2))
        encoder.add(layers.Activation(tf.nn.relu))
        encoder.add(layers.MaxPooling2D((2, 2)))
    encoder.add(layers.Flatten())

    # For Koch et al.-style siamese network
    # encoder.add(layers.Dense(2048, activation=tf.nn.sigmoid))
    
    return encoder

# def _encoder_model(input_shape):
#     encoder = tf.keras.models.Sequential([
#         layers.Input(input_shape),
#         layers.Conv2D(3, (1,1), activation='relu'),  # Make multi-channel input compatible with Inception                    
#         hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4", trainable=False),
#         layers.Dense(512),
#         layers.BatchNormalization(),
#     ], name="encoder")
#     encoder.summary()
#     return encoder

def _euclidean_distance(tensors):
    sum_square = tf.math.reduce_sum(tf.math.squared_difference(tensors[0], tensors[1]), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def _siamese_model(input_shape):
    query = layers.Input(input_shape, name="query")
    support = layers.Input(input_shape, name="support")

    encoder = _encoder_model(input_shape)
    embeddings = [encoder(query), encoder(support)]

    output = layers.Lambda(_euclidean_distance)(embeddings)
    
    # For Koch et al.-style siamese network
    # output = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))(embeddings)
    # output = layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True)(output)

    return tf.keras.Model(inputs=[query, support], outputs=output)


def _random_pair_indices(replicates, max_pairs):
    match_pairs = itertools.product(range(3), itertools.combinations(range(replicates), 2))
    match_pairs = [((ac, r1), (ac, r2)) for (ac, (r1, r2)) in match_pairs]

    mismatch_pairs = itertools.product(
        itertools.combinations(range(3), 2),
        itertools.product(range(replicates), repeat=2),
    )
    mismatch_pairs = [((ac1, r1), (ac2, r2)) for ((ac1, ac2), (r1, r2)) in mismatch_pairs]

    # Interleave positive and negative examples, flattening pairs into a single list
    class_num_pairs = min(len(match_pairs), len(mismatch_pairs), max_pairs // 2)
    pairs = [pair for pos_and_neg in zip(random.sample(match_pairs, class_num_pairs), random.sample(mismatch_pairs, class_num_pairs)) for pair in pos_and_neg]
    print(pairs)  
    # "Split" pairs into a list of query indices and a list of support indices
    return zip(*pairs), [1, 0] * class_num_pairs


def _variant_to_training_pairs(features, original_label, image_shape, replicates, max_pairs=8):
    possible_matches = 3 * replicates * (replicates - 1) // 2
    possible_mismatches = ((3 * replicates) * (3 * replicates - 1) // 2) - possible_matches
    pairs_per_class = min(possible_matches, possible_mismatches, max_pairs // 2)

    images = tf.ensure_shape(tf.image.convert_image_dtype(features["sim/images"], dtype=tf.float32), (3, replicates) + image_shape)
    flat_images = tf.reshape(images, (-1,) + image_shape)

    # We want to ensure we get a different set of images for every variant
    replicate_labels = tf.expand_dims(tf.repeat(range(3), repeats=replicates), axis=-1) # (number of images, 1)
    adjacency = tf.math.equal(replicate_labels, tf.transpose(replicate_labels))

    match_pairs = tf.where(adjacency)
    match_pairs = tf.boolean_mask(match_pairs, tf.math.less(match_pairs[:,0], match_pairs[:,1]))
    match_pairs = tf.random.shuffle(match_pairs)[:pairs_per_class]

    mismatch_pairs = tf.where(tf.math.logical_not(adjacency))
    mismatch_pairs = tf.boolean_mask(mismatch_pairs, tf.math.less(mismatch_pairs[:,0], mismatch_pairs[:,1]))
    mismatch_pairs = tf.random.shuffle(mismatch_pairs)[:pairs_per_class]

    # Interleave match-mismatch pairs into a single tensor
    pairs = tf.dynamic_stitch([range(0, 2*pairs_per_class, 2), range(1, 2*pairs_per_class, 2)], [match_pairs, mismatch_pairs])

    image_tensors = {
        "query": tf.gather(flat_images, pairs[:,0]),
        "support": tf.gather(flat_images, pairs[:,1]),
    }
    label_tensor = tf.tile(tf.constant([1, 0]), [pairs_per_class])
   
    # Construct data to permit "flat_map" and thus more flexible batching downstream
    return tf.data.Dataset.from_tensor_slices((image_tensors, label_tensor))


def distance_accuracy(y_true, y_pred, threshold=0.5):
    """Compute classification accuracy with a fixed threshold on distances."""
    return tf.keras.backend.mean(tf.math.equal(y_true, tf.cast(y_pred < threshold, y_true.dtype)), axis=-1)


def train(params, tfrecords_path: str, model_path: str):
    image_shape, replicates = _extract_metadata_from_first_example(tfrecords_path)

    model = _siamese_model(image_shape)
    model.summary()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=params.learning_rate,
        decay_steps=10000,  # e.g., set to the number of steps per epoch
        decay_rate=0.99,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss=tfa.losses.contrastive_loss,  # For models that just take euclidean distance...
        metrics=[distance_accuracy],
        # loss=tf.keras.losses.binary_crossentropy,  # For Koch et al.-style siamese networks
        # metrics=["binary_accuracy"],
    )

    example_dataset = load_example_dataset(
        tfrecords_path,
        with_label=True,
        with_simulations=True,
    )
    variant_to_training_pairs = functools.partial(_variant_to_training_pairs, image_shape=image_shape, replicates=replicates, max_pairs=sys.maxsize)
    train_dataset = (
        example_dataset.shuffle(1000, reshuffle_each_iteration=True).flat_map(variant_to_training_pairs).batch(32)
    )

    # TODO: Reserve some training data for validation
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)

    checkpoint_filepath = os.path.join(params.tempdir, "checkpoint")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, save_weights_only=True, monitor="loss", mode="min", save_best_only=True
    )

    callbacks=[checkpoint_callback]#[early_stopping, checkpoint_callback]
    
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
    query_images = tf.stack([tf.image.convert_image_dtype(features["image"], dtype=tf.float32)] * 3)
    support_images = tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32)

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
        #genotype_probabilities = tf.reshape(model.predict(images), (-1, 3))
        genotype_distances = tf.reshape(model.predict(images), (-1, 3))
        genotype_probabilities = tf.nn.softmax(-genotype_distances, axis=1)  # For models that produce distances

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


def _variant_to_visualize_pairs(features, original_label, image_shape, replicates):
    query_images = tf.image.convert_image_dtype(features["image"], dtype=tf.float32) #tf.stack([tf.image.convert_image_dtype(features["image"], dtype=tf.float32)] * 3 * replicates)
    support_images = tf.image.convert_image_dtype(features["sim/images"], dtype=tf.float32) #tf.reshape(tf.image.convert_image_dtype(features["sim/images"], dtype=tf.float32), (-1,) + image_shape)

    image_tensors = {
        "query": query_images,
        "support": support_images,
    }
    return image_tensors, original_label, features["variant/encoded"]

def visualize_embeddings(model, dataset, image_shape=None, replicates=None):
    if isinstance(model, str):
        model = tf.keras.models.load_model(
            model,
            custom_objects={"distance_accuracy": distance_accuracy, "contrastive_loss": tfa.losses.contrastive_loss},
        )
    assert isinstance(model, tf.keras.Model), "Valid model or path not provided"

    if isinstance(dataset, str):
        image_shape, replicates = _extract_metadata_from_first_example(dataset)
        dataset = load_example_dataset(dataset, with_label=True, with_simulations=True)

    model.summary()
    # encoder = model.get_layer("encoder")
    # encoder.summary()

    variant_to_visualize_pairs = functools.partial(_variant_to_visualize_pairs, image_shape=image_shape, replicates=replicates)
    test_dataset = dataset.map(variant_to_visualize_pairs)

    for images, label, encoded_variant in test_dataset:
        
        rows = []
        for (r1, r2) in itertools.combinations(range(3 * replicates), 2):
            p = model.predict({ 
                "query": tf.reshape(images["support"][r1 // replicates][r1 % replicates], (1,) + image_shape),
                "support": tf.reshape(images["support"][r2 // replicates][r2 % replicates], (1,) + image_shape),
            })
            print(r1, r2, p)

            rows.append(
                pd.DataFrame(
                    dict(AC1=r1 // replicates, AC2=r2 // replicates, SIM=p[0])
                )
            )

        table = pd.concat(rows, ignore_index=True)

        print(table.groupby(["AC1", "AC2"]).mean())
    #   print(model.predict(images))
    #     query_embeddings = encoder.predict(tf.reshape(images["query"], (1,) + image_shape))
    #     support_embeddings = encoder.predict(tf.reshape(images["support"], (-1,) + image_shape))
    #     print(distance.cdist(query_embeddings, support_embeddings))
        