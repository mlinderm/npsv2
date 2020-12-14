import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from .images import load_example_dataset, _extract_metadata_from_first_example, _features_variant

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

def _inceptionv3_encoder(input_shape):
    assert tf.keras.backend.image_data_format() == "channels_last"

    base_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape[:-1] + (3,), pooling="avg")
    base_model.trainable = True
    
    encoder = tf.keras.models.Sequential([
        layers.Input(input_shape),
        layers.Conv2D(3, (1,1), activation='tanh'),  # Make multi-channel input compatible with Inception (input [-1, 1])
        base_model,
        layers.Dense(512),
        layers.BatchNormalization(),
    ], name="encoder")
    return encoder

def _variant_to_allsim_training_triples(features, original_label):
    # TODO: Make this a function of the number of replicates
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
    # TODO: Make this a function of the number of replicates
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
 

def _variant_to_test_triples(features, original_label):
    query_images = tf.image.convert_image_dtype(features["image"], dtype=tf.float32)
    support_images = tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32)
   
    image_tensors = {
        "query": query_images,
        "support": support_images,
    }
    return image_tensors, original_label, features["variant/encoded"]

class SiameseGenotyper:
    def __init__(self, input_shape, replicates):
        self.input_shape = input_shape
        self.replicates = replicates
    
    def train_input(self, filenames, shuffle=1000, batch_size=24):
        example_dataset = load_example_dataset(filenames, with_label=True, with_simulations=True)

        example_dataset = example_dataset.shuffle(1000, reshuffle_each_iteration=True)
        #example_dataset = example_dataset.flat_map(_variant_to_allsim_training_triples)
        example_dataset = example_dataset.flat_map(_variant_to_real_training_triples)
        example_dataset = example_dataset.batch(batch_size)
        
        return example_dataset

    def test_input(self, filenames):
        example_dataset = load_example_dataset(filenames, with_label=True, with_simulations=True)
        return example_dataset.map(_variant_to_test_triples).batch(1)

    def validation_input(self, filenames, batch_size=24):
        example_dataset = load_example_dataset(filenames, with_label=True, with_simulations=True)

        # Strip off variant description and batch the same way as the training data
        return example_dataset.map(_variant_to_test_triples).map(lambda image_tensors, original_label, _: (image_tensors, original_label)).batch(batch_size)

    def model_fn(self, params=None, model_path:str=None):
        encoder = _inceptionv3_encoder(self.input_shape)
        
        query = layers.Input(self.input_shape, name="query")
        query_embeddings = encoder(query)
        
        support = layers.Input((3,) + self.input_shape, name="support")
        support_embeddings = layers.TimeDistributed(encoder)(support)
        
        def _variant_distances(tensors):
            query, support = tensors
            return tf.squeeze(tf.map_fn(_cdist, (tf.expand_dims(query, axis=1), support), dtype=tf.dtypes.float32), axis=1)

        distances = layers.Lambda(_variant_distances, name="distances")([query_embeddings, support_embeddings])
        output = layers.Lambda(lambda x: tf.nn.softmax(-x, axis=1), name="genotypes")(distances) # Convert distance to probability

        model = tf.keras.Model(inputs=[query, support], outputs=[output, distances], name="SiameseGenotyper")
        if model_path:
            encoder.load_weights(model_path)

        if params:
            optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
            model.compile(
                optimizer=optimizer,
                loss={"genotypes": "sparse_categorical_crossentropy"},
                metrics={"genotypes": ["sparse_categorical_accuracy"]},
            )

        return model

    def save_model(self, model, model_path: str):
        # We only save the encoder weights
        encoder = model.get_layer("encoder")
        encoder.save_weights(model_path)