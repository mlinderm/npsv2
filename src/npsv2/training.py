import datetime, functools, itertools, logging, os, random, sys
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from scipy.spatial import distance
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from .images import load_example_dataset, _extract_metadata_from_first_example, _features_variant
from . import npsv2_pb2
from . import models



# def _encoder_model(input_shape):
#     encoder = tf.keras.models.Sequential([layers.Input(input_shape)], name="encoder")
#     for i in range(4):
#         encoder.add(layers.Conv2D(64, (3, 3), padding="same"))
#         encoder.add(layers.BatchNormalization())
#         encoder.add(layers.Dropout(0.2))
#         encoder.add(layers.Activation(tf.nn.relu))
#         encoder.add(layers.MaxPooling2D((2, 2)))
#     encoder.add(layers.Flatten())

#     # For Koch et al.-style siamese network
#     # encoder.add(layers.Dense(2048, activation=tf.nn.sigmoid))
    
#    return encoder

# def _encoder_model(input_shape):
#     encoder = tf.keras.models.Sequential([
#         layers.Input(input_shape),
#         layers.Conv2D(3, (1,1), activation='relu'),  # Make multi-channel input compatible with Inception                    
#         hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4", trainable=False),
#         layers.Dense(512),
#         layers.BatchNormalization(),
#     ], name="encoder")
#     return encoder

def _encoder_model(input_shape):
    assert tf.keras.backend.image_data_format() == "channels_last"

    base_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape[:-1] + (3,), pooling="avg")
    base_model.trainable = True #False
    
    # Trying to fine tune model with imagenet weights but 4 input channels, but am not successful. All layers mismatch...
    # base_model = tf.keras.applications.InceptionV3(include_top=False, weights=None, input_shape=input_shape, pooling="avg")
    # weights_path = tf.keras.utils.get_file(
    #       'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #       'https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #       cache_subdir='models',
    #       file_hash='bcbd6486424b2319ff4ef7d526e38f63')
    # base_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    # base_model.trainable = False

    encoder = tf.keras.models.Sequential([
        layers.Input(input_shape),
        #layers.Conv2D(3, (1,1), activation='relu'),  # Make multi-channel input compatible with Inception     
        layers.Conv2D(3, (1,1), activation='tanh'),  # Make multi-channel input compatible with Inception (input [-1, 1])
        base_model,
        layers.Dense(512),
        layers.BatchNormalization(),
    ], name="encoder")
    return encoder

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

def _cdist(tensors, squared: bool = False):
    # https://github.com/tensorflow/addons/blob/81529ff7dd246f7575338b8cfe65784b0cc8a502/tensorflow_addons/losses/metric_learning.py#L21-L67
    query_features, genotype_features = tensors

    distances_squared = tf.math.add(
        tf.math.reduce_sum(tf.math.square(query_features), axis=[1], keepdims=True),
        tf.math.reduce_sum(tf.math.square(tf.transpose(genotype_features)), axis=[0], keepdims=True),
    ) - 2.0 * tf.matmul(query_features, tf.transpose(genotype_features))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    distances_squared = tf.math.maximum(distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = tf.math.less_equal(distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        distances = distances_squared
    else:
        distances = tf.math.sqrt(
            distances_squared + tf.cast(error_mask, dtype=tf.dtypes.float32) * tf.keras.backend.epsilon()
        )

    # Undo conditionally adding 1e-16.
    distances = tf.math.multiply(
        distances,
        tf.cast(tf.math.logical_not(error_mask), dtype=tf.dtypes.float32),
    )
    return distances



def _genotype_model(input_shape):
    encoder = _encoder_model(input_shape)
    encoder.summary()
    embeddings_shape = encoder.output_shape
    # print(embeddings_shape)
    
    query = layers.Input(input_shape, name="query")
    query_embeddings = encoder(query)
       
    support = layers.Input((3,) + input_shape, name="support")
    support_embeddings = layers.TimeDistributed(encoder)(support)
    #support_embeddings = layers.Lambda(lambda x: tf.reshape(encoder(tf.reshape(x, (-1,) + input_shape)), (-1, 3, embeddings_shape[-1])))(support)

    def _variant_distances(tensors):
        query, support = tensors
        return tf.squeeze(tf.map_fn(_cdist, (tf.expand_dims(query, axis=1), support), dtype=tf.dtypes.float32), axis=1)

    distances = layers.Lambda(_variant_distances)([query_embeddings, support_embeddings])
    output = layers.Lambda(lambda x: tf.nn.softmax(-x, axis=1))(distances) # Convert distance to probability

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

def _variant_to_training_triples(features, original_label):
    # query_images = tf.stack([tf.image.convert_image_dtype(features["image"], dtype=tf.float32)]*4)
    # support_images = tf.stack([
    #     tf.image.convert_image_dtype(features["sim/images"][:,1], dtype=tf.float32),
    #     tf.image.convert_image_dtype(features["sim/images"][:,2], dtype=tf.float32),
    #     tf.image.convert_image_dtype(features["sim/images"][:,3], dtype=tf.float32),
    #     tf.image.convert_image_dtype(features["sim/images"][:,4], dtype=tf.float32),
    # ])

    # image_tensors = {
    #     "query": query_images,
    #     "support": support_images,
    # }
    
    # return tf.data.Dataset.from_tensor_slices((image_tensors, tf.repeat(original_label, 4)))
    # query_images = tf.expand_dims(tf.image.convert_image_dtype(features["image"], dtype=tf.float32), axis=0)
    # support_images = tf.expand_dims(tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32), axis=0)

    # image_tensors = {
    #     "query": query_images,
    #     "support": support_images,
    # }
    # return tf.data.Dataset.from_tensor_slices((image_tensors, original_label))
    # query_images = tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32)
    # support_images = tf.stack([tf.image.convert_image_dtype(features["sim/images"][:,1], dtype=tf.float32)]*3)

    # image_tensors = {
    #     "query": query_images,
    #     "support": support_images,
    # }
    # return tf.data.Dataset.from_tensor_slices((image_tensors, tf.constant([0, 1, 2])))
    query_images = tf.tile(
        tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32),
        [4, 1, 1, 1]
    )
    support_images = tf.stack([
        tf.image.convert_image_dtype(features["sim/images"][:,1], dtype=tf.float32),
        tf.image.convert_image_dtype(features["sim/images"][:,1], dtype=tf.float32),
        tf.image.convert_image_dtype(features["sim/images"][:,1], dtype=tf.float32),
        tf.image.convert_image_dtype(features["sim/images"][:,2], dtype=tf.float32),
        tf.image.convert_image_dtype(features["sim/images"][:,2], dtype=tf.float32),
        tf.image.convert_image_dtype(features["sim/images"][:,2], dtype=tf.float32),
        tf.image.convert_image_dtype(features["sim/images"][:,3], dtype=tf.float32),
        tf.image.convert_image_dtype(features["sim/images"][:,3], dtype=tf.float32),
        tf.image.convert_image_dtype(features["sim/images"][:,3], dtype=tf.float32),
        tf.image.convert_image_dtype(features["sim/images"][:,4], dtype=tf.float32),
        tf.image.convert_image_dtype(features["sim/images"][:,4], dtype=tf.float32),
        tf.image.convert_image_dtype(features["sim/images"][:,4], dtype=tf.float32),
    ])
    image_tensors = {
        "query": query_images,
        "support": support_images,
    }
    return tf.data.Dataset.from_tensor_slices((image_tensors, tf.constant([0, 1, 2]*4, dtype=tf.int64)))

def _variant_to_real_training_triples(features, original_label):
    query_images = tf.stack([tf.image.convert_image_dtype(features["image"], dtype=tf.float32)]*4)
    support_images = tf.stack([
        tf.image.convert_image_dtype(features["sim/images"][:,1], dtype=tf.float32),
        tf.image.convert_image_dtype(features["sim/images"][:,2], dtype=tf.float32),
        tf.image.convert_image_dtype(features["sim/images"][:,3], dtype=tf.float32),
        tf.image.convert_image_dtype(features["sim/images"][:,4], dtype=tf.float32),
    ])

    image_tensors = {
        "query": query_images,
        "support": support_images,
    }
    
    return tf.data.Dataset.from_tensor_slices((image_tensors, tf.repeat(original_label, 4)))


def distance_accuracy(y_true, y_pred, threshold=0.5):
    """Compute classification accuracy with a fixed threshold on distances."""
    return tf.keras.backend.mean(tf.math.equal(y_true, tf.cast(y_pred < threshold, y_true.dtype)), axis=-1)


def train(params, tfrecords_paths, model_path: str):
    if isinstance(tfrecords_paths, str):
        tfrecords_paths = [tfrecords_paths]
    assert len(tfrecords_paths) > 0

    image_shape, replicates = _extract_metadata_from_first_example(tfrecords_paths[0])
    genotyper = models.JointEmbeddingsModel(image_shape, replicates)
    
    dataset = load_example_dataset(tfrecords_paths, with_label=True, with_simulations=True)
    genotyper.fit(dataset, validation_dataset=dataset, epochs=params.epochs, learning_rate=params.learning_rate)
    
    logging.info("Saving model in: %s", model_path)
    genotyper.save(model_path)


def _variant_to_test_pairs(features, original_label):
    query_images = tf.stack([tf.image.convert_image_dtype(features["image"], dtype=tf.float32)] * 3)
    support_images = tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32)

    image_tensors = {
        "query": query_images,
        "support": support_images,
    }
    return image_tensors, original_label, features["variant/encoded"]

def _variant_to_test_triples(features, original_label):
    query_images = tf.image.convert_image_dtype(features["image"], dtype=tf.float32)
    support_images = tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32)
   
    image_tensors = {
        "query": tf.expand_dims(query_images, axis=0),
        "support": tf.expand_dims(support_images, axis=0),
    }
    return image_tensors, original_label, features["variant/encoded"]


def evaluate_model(params, tfrecords_paths, model_path: str):
    if isinstance(tfrecords_paths, str):
        tfrecords_paths = [tfrecords_paths]
    assert len(tfrecords_paths) > 0

    image_shape, replicates = _extract_metadata_from_first_example(tfrecords_paths[0])
    genotyper = models.JointEmbeddingsModel(image_shape, replicates, model_path=model_path)
    predict_fn = genotyper.make_predict()

    rows = []
    for features, original_label in load_example_dataset(tfrecords_paths, with_label=True, with_simulations=True):
        # Extract metadata for the variant
        variant_proto = npsv2_pb2.StructuralVariant.FromString(features.pop("variant/encoded").numpy())

        # Predict genotype
        dataset = tf.data.Dataset.from_tensors((features, original_label))
        genotypes, distances, *_  = predict_fn(dataset)
        
        print(variant_proto, genotypes, distances)
        # if tf.math.argmax(genotypes, axis=1) == label and label == 2:
        #     print(variant_proto, genotypes, distances)
        #     errors += 1
        # assert errors < 10


        # Construct the DataFrame rows
        rows.append(pd.DataFrame({
            "SVLEN": variant_proto.svlen,
            "LABEL": original_label,
            "AC": tf.math.argmax(genotypes, axis=1),
        }))

    table = pd.concat(rows, ignore_index=True)

    table["AC"] = pd.Categorical(table["AC"], categories=[0, 1, 2])
    table["LABEL"] = pd.Categorical(table["LABEL"], categories=[0, 1, 2])
    table["MATCH"] = table.LABEL == table.AC
    
    return table



def _variant_to_visualize_pairs(features, original_label, image_shape, replicates):
    query_images = tf.image.convert_image_dtype(features["image"], dtype=tf.float32) #tf.stack([tf.image.convert_image_dtype(features["image"], dtype=tf.float32)] * 3 * replicates)
    support_images = tf.image.convert_image_dtype(features["sim/images"], dtype=tf.float32) #tf.reshape(tf.image.convert_image_dtype(features["sim/images"], dtype=tf.float32), (-1,) + image_shape)

    image_tensors = {
        "query": query_images,
        "support": support_images,
    }
    return image_tensors, original_label, features["variant/encoded"]

def visualize_embeddings(tfrecords_paths, model_path, labels=["Hom. ref.", "Het.", "Hom. alt."], colors=["red", "green", "blue"]):
    # Setup for plotting
    import matplotlib.pyplot as plt
    from sklearn import decomposition
    from sklearn.preprocessing import StandardScaler
    
    pca = decomposition.PCA(n_components=2)

    if isinstance(tfrecords_paths, str):
        tfrecords_paths = [tfrecords_paths]
    assert len(tfrecords_paths) > 0

    image_shape, replicates = _extract_metadata_from_first_example(tfrecords_paths[0])

    # Extract underlying encoder
    genotyper = models.JointEmbeddingsModel(image_shape, replicates, model_path=model_path)
    model = genotyper._model
    encoder = model.get_layer("encoder")
    
    example_dataset = load_example_dataset(tfrecords_paths, with_label=True, with_simulations=True)
    for i, (features, original_label) in enumerate(example_dataset):
        query_embeddings = encoder.predict(tf.expand_dims(tf.image.convert_image_dtype(features["image"], dtype=tf.float32), axis=0))
        
        support_images = tf.reshape(tf.image.convert_image_dtype(features["sim/images"], dtype=tf.float32), (-1,) + image_shape)
        support_embeddings = encoder.predict(support_images)

        embeddings = tf.concat([support_embeddings, query_embeddings], axis=0)
        embeddings_std = StandardScaler().fit_transform(embeddings.numpy())
        embeddings_2d = pca.fit_transform(embeddings_std)
        
        plt, ax = plt.subplots()
        for ac in range(3):
            ac_embeddings = embeddings_2d[ac*replicates:(ac+1)*replicates,:]
            ax.scatter(ac_embeddings[:,0], ac_embeddings[:,1], c=colors[ac], label=labels[ac])
        
        real_embeddings = embeddings_2d[3 * replicates, :]
        ax.scatter(real_embeddings[0], real_embeddings[1], marker="x", c="black", label="Real")
        ax.legend()
        plt.savefig(f"variant{i}.png")
    
        