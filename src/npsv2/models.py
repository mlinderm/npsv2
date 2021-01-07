import datetime, logging, os, tempfile
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2

from .util.config import Config, merge_config
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


def _inceptionv3_encoder(input_shape, normalize=False):
    assert tf.keras.backend.image_data_format() == "channels_last"

    base_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape[:-1] + (3,), pooling="avg")
    #base_model = tf.keras.applications.InceptionV3(include_top=False, weights=None, input_shape=input_shape, pooling="avg")
    base_model.trainable = True
    
    encoder = tf.keras.models.Sequential([
        layers.Input(input_shape),
        layers.Conv2D(3, (1,1), activation='tanh'),  # Make multi-channel input compatible with Inception (input [-1, 1])
        base_model,
        layers.Dense(512),
        layers.BatchNormalization(),
    ], name="encoder")
    if normalize:
        encoder.add(layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))) # L2 normalize embeddings
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

        example_dataset = example_dataset.shuffle(shuffle, reshuffle_each_iteration=True)
        example_dataset = example_dataset.flat_map(_variant_to_allsim_training_triples)
        #example_dataset = example_dataset.flat_map(_variant_to_real_training_triples)
        example_dataset = example_dataset.batch(batch_size)
        
        return example_dataset

    def test_input(self, filenames):
        example_dataset = load_example_dataset(filenames, with_label=True, with_simulations=True)
        return example_dataset.map(_variant_to_test_triples).batch(1)

    def validation_input(self, filenames, batch_size=24):
        example_dataset = load_example_dataset(filenames, with_label=True, with_simulations=True)

        # Strip off variant description and batch the same way as the training data
        return example_dataset.map(_variant_to_test_triples).map(lambda image_tensors, original_label, _: (image_tensors, original_label)).batch(batch_size)
        #return None

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

class GenotypingModelConfig(Config):
    temp_dir: str = tempfile.gettempdir()
    log_dir: str = None
    learning_rate: float = 0.004
    epochs: int = 20

class GenotypingModel:    
    def summary(self):
        self._model.summary()

    def save(self, model_path: str):
        # We only save the encoder weights
        encoder = self._model.get_layer("encoder")
        encoder.save_weights(model_path)

    def _fit(self, config, training_dataset, validation_dataset=None, **kwargs):
        checkpoint_filepath = os.path.join(config.tempdir, "checkpoint")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=True, monitor="loss", mode="min", save_best_only=True
        )
        #early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)

        callbacks=[checkpoint_callback] #,early_stopping]
    
        if config.log_dir:
            log_dir = os.path.join(config.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            logging.info("Logging TensorBoard data to: %s", log_dir)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            callbacks.append(tensorboard_callback)

        self._model.fit(
            training_dataset,
            validation_data=validation_dataset,
            epochs=config.epochs,
            callbacks=callbacks,
        )
    
        # Reload best checkpoint
        self._model.load_weights(checkpoint_filepath)


class TripletModelConfig(GenotypingModelConfig):
    variants_per_batch: int = 8
    shuffle: int = 1000


class TripletModel(GenotypingModel):
    def __init__(self, image_shape, replicates, model_path: str=None, **kwargs):
        self.config = merge_config(TripletModelConfig(), kwargs)
        
        self.image_shape = image_shape
        self.replicates = replicates

        encoder = _inceptionv3_encoder(image_shape, normalize=True)       
        if model_path:
            encoder.load_weights(model_path)
        
        self._model = TripletModel._model_from_encoder(encoder, image_shape, replicates)

    @classmethod
    def _model_from_encoder(cls, encoder, image_shape, replicates):
        embedding_shape = encoder.output_shape
        
        support = layers.Input((3, replicates) + image_shape, name="support")
        support_embeddings = layers.Reshape((3 * replicates,) + image_shape)(support)
        support_embeddings = layers.TimeDistributed(encoder, name="support_embeddings")(support_embeddings)
       
        query = layers.Input(image_shape, name="query")
        query_embeddings = encoder(query)
        
        def _query_distances(tensors):
            query_embeddings, support_batch = tensors

            # Make query embeddings have shape (batch, 1, ...) so we can map across batch to compute
            # distances between query example and all support examples
            support_distances = tf.map_fn(_cdist, (tf.expand_dims(query_embeddings, axis=1), support_batch), dtype=tf.dtypes.float32)
            support_distances = tf.squeeze(support_distances, axis=1)  # Remove unit dimension for query

            # An alternate approach could use the genotype "cluster" means instead of cluster minimum
            cluster_distances = tf.math.reduce_min(tf.reshape(support_distances, (-1, 3, replicates)), axis=-1)
            return cluster_distances

        distances = layers.Lambda(_query_distances, name="distances")([query_embeddings, support_embeddings])
        
        # Convert distance to probability
        genotypes = layers.Lambda(lambda x: tf.nn.softmax(-x, axis=-1), name="genotypes")(distances) 

        return tf.keras.Model(inputs=[query, support], outputs=[genotypes, distances, support_embeddings])

    
    def _train_input(self, config, dataset):
        def _variant_to_training(index, element):
            features, original_label = element
            triplet_labels = range(3*index, 3*(index + 1))
            return ({ 
                "query": tf.image.convert_image_dtype(features["image"], dtype=tf.float32),
                "support": tf.image.convert_image_dtype(features["sim/images"], dtype=tf.float32),
            }, {
                "support_embeddings": tf.repeat(triplet_labels, repeats=self.replicates),
                "genotypes": original_label,
            })
        
        return dataset.shuffle(config.shuffle, reshuffle_each_iteration=True).enumerate().map(_variant_to_training).batch(config.variants_per_batch)
        

    def fit(self, training_dataset, validation_dataset=None, **kwargs):
        config = merge_config(self.config, kwargs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        
        def _loss_wrapper(y_true, y_pred):
            # "Flatten" batch size (assuming unique label for each variant)
            embedding_shape = tf.shape(y_pred)
            y_true = tf.reshape(y_true, (-1,))
            y_pred = tf.reshape(y_pred, (-1, embedding_shape[-1]))
            return tfa.losses.triplet_semihard_loss(y_true, y_pred)

        self._model.compile(
            optimizer=optimizer,
            #loss={ "support_embeddings": _loss_wrapper, "genotypes": "sparse_categorical_crossentropy" },
            loss={ "genotypes": "sparse_categorical_crossentropy" },
            metrics={ "genotypes": ["sparse_categorical_accuracy"] },
        )
                
        self._model.fit(
            self._train_input(config, training_dataset),
            epochs=config.epochs,
        )

    def _test_input(self, dataset):
        def _variant_to_test(features, original_label):
            return {
                "query": tf.image.convert_image_dtype(features["image"], dtype=tf.float32),
                "support": tf.expand_dims(tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32), axis=1),
            }, original_label

        return dataset.map(_variant_to_test).batch(1)


    def make_predict(self, **kwargs):
        # Build a simplified version of the model that expects only one replicate
        encoder = self._model.get_layer("encoder")
        model = TripletModel._model_from_encoder(encoder, self.image_shape, replicates=1)
        
        return lambda dataset: model.predict(self._test_input(dataset))

    def predict(self, dataset, **kwargs):
        predict_fn = self.make_predict(**kwargs)
        return predict_fn(dataset)


class JointEmbeddingsModelConfig(GenotypingModelConfig):
    variants_per_batch: int = 8
    shuffle: int = 1000


class JointEmbeddingsModel(GenotypingModel):
    def __init__(self, image_shape, replicates, model_path: str=None, **kwargs):
        self.config = merge_config(JointEmbeddingsModelConfig(), kwargs)
        
        self.image_shape = image_shape
        self.replicates = replicates

        encoder = _inceptionv3_encoder(image_shape, normalize=True)       
        if model_path:
            encoder.load_weights(model_path)
        
        self._model = JointEmbeddingsModel._model_from_encoder(encoder, image_shape, replicates)
    
    @classmethod
    def _model_from_encoder(cls, encoder, image_shape, replicates):
        embedding_shape = encoder.output_shape
        
        support = layers.Input((3, replicates) + image_shape, name="support")
        support_embeddings = layers.Reshape((3 * replicates,) + image_shape)(support)
        support_embeddings = layers.TimeDistributed(encoder, name="support_embeddings")(support_embeddings)
       
        query = layers.Input(image_shape, name="query")
        query_embeddings = encoder(query)
        
        def _query_distances(tensors):
            query_embeddings, support_batch = tensors

            # Make query embeddings have shape (batch, 1, ...) so we can map across batch to compute
            # distances between query example and all support examples
            support_distances = tf.map_fn(_cdist, (tf.expand_dims(query_embeddings, axis=1), support_batch), dtype=tf.dtypes.float32)
            support_distances = tf.squeeze(support_distances, axis=1)  # Remove unit dimension for query

            return support_distances

        support_distances = layers.Lambda(_query_distances, name="support_distances")([query_embeddings, support_embeddings])
        
        # An alternate approach could use the genotype "cluster" means instead of cluster minimum
        distances = layers.Lambda(lambda x: tf.math.reduce_min(tf.reshape(x, (-1, 3, replicates)), axis=-1), name="distances")(support_distances)
        
        # Convert distance to probability
        genotypes = layers.Lambda(lambda x: tf.nn.softmax(-x, axis=-1), name="genotypes")(distances) 

        return tf.keras.Model(inputs=[query, support], outputs=[genotypes, distances, support_distances])

    def _train_input(self, config, dataset):
        def _variant_to_training(features, original_label):
            contrastive_labels = tf.one_hot(original_label, depth=3, dtype=tf.float32)

            return ({ 
                "query": tf.image.convert_image_dtype(features["image"], dtype=tf.float32),
                "support": tf.image.convert_image_dtype(features["sim/images"], dtype=tf.float32),
            }, {
                "support_distances": tf.repeat(contrastive_labels, repeats=self.replicates),
                "genotypes": original_label,
            })
        
        return dataset.shuffle(config.shuffle, reshuffle_each_iteration=True).map(_variant_to_training).batch(config.variants_per_batch)
        

    def fit(self, training_dataset, validation_dataset=None, **kwargs):
        config = merge_config(self.config, kwargs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        
        def _loss_wrapper(y_true, y_pred):
            # "Flatten" batch size
            y_true = tf.reshape(y_true, (-1,))
            y_pred = tf.reshape(y_pred, (-1,))
            return tfa.losses.contrastive_loss(y_true, y_pred)

        self._model.compile(
            optimizer=optimizer,
            loss={ "support_distances": _loss_wrapper },
            metrics={ "genotypes": ["sparse_categorical_accuracy"] },
        )
                
        self._model.fit(
            self._train_input(config, training_dataset),
            epochs=config.epochs,
        )

    def _test_input(self, dataset):
        def _variant_to_test(features, original_label):
            return {
                "query": tf.image.convert_image_dtype(features["image"], dtype=tf.float32),
                "support": tf.expand_dims(tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32), axis=1),
            }, original_label

        return dataset.map(_variant_to_test).batch(1)


    def make_predict(self, **kwargs):
        # Build a simplified version of the model that expects only one replicate
        encoder = self._model.get_layer("encoder")
        model = JointEmbeddingsModel._model_from_encoder(encoder, self.image_shape, replicates=1)
        
        return lambda dataset: model.predict(self._test_input(dataset))

    def predict(self, dataset, **kwargs):
        predict_fn = self.make_predict(**kwargs)
        return predict_fn(dataset)