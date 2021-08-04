import datetime, logging, os, tempfile
from omegaconf import OmegaConf
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from google.protobuf import descriptor_pb2

from .images import load_example_dataset, _extract_metadata_from_first_example, _features_variant
from . import npsv2_pb2

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

def _query_distances(tensors):
    query_embeddings, support_batch = tensors

    # Make query embeddings have shape (batch, 1, ...) so we can map across batch to compute
    # distances between query example and all support examples
    support_distances = tf.map_fn(_cdist, (tf.expand_dims(query_embeddings, axis=1), support_batch), fn_output_signature=tf.dtypes.float32)
    support_distances = tf.squeeze(support_distances, axis=1)  # Remove unit dimension for query

    return support_distances


def _base_model(input_shape, weights="imagenet", trainable=True):
    assert tf.keras.backend.image_data_format() == "channels_last"

    if weights is not None and input_shape[-1] != 3:
        # imagenet weights require a 3-channel input image. To enable us to use more channels, we construct a dummy model
        # with 3-channel image and copy those weights where possible into a network with the desired size
        base_model = tf.keras.applications.InceptionV3(include_top=False, weights=None, input_shape=input_shape, pooling="avg")

        # Only the first convolution layer needs to have different weights
        src_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape[:-1] + (3,), pooling="avg")
        for i, (src_layer, dst_layer) in enumerate(zip(src_model.layers, base_model.layers)):
            if i == 1:
                # Replicate mean of existing first convolutional layer (assumes "channels_last"). Motivated by
                # https://arxiv.org/pdf/1608.00859.pdf and https://stackoverflow.com/a/62631570
                kernels, *other_weights = src_layer.get_weights()
                new_kernels = tf.repeat(tf.reduce_mean(kernels, axis=-2, keepdims=True) * (3/input_shape[-1]), input_shape[-1], axis=-2)
                dst_layer.set_weights([new_kernels] + other_weights)
                dst_layer.trainable = True # New layer should default to trainable
            else:
                dst_layer.set_weights(src_layer.get_weights())
                dst_layer.trainable = trainable
    else:
        base_model = tf.keras.applications.InceptionV3(include_top=False, weights=weights, input_shape=input_shape, pooling="avg")
        base_model.trainable = trainable
    
    return base_model



# Adapted from:
# https://github.com/google-research/google-research/blob/e2308c7593eda306daab40db07930a2d5132255b/supcon/models.py#L26
def _contrastive_encoder(input_shape, weights=None, base_trainable=True, normalize_embedding=True, stop_gradient_before_projection=False, projection_size=[128], batch_normalize_projection=False, normalize_projection=True):
    assert tf.keras.backend.image_data_format() == "channels_last"

    image = layers.Input(input_shape) 
    
    # Set trainable to False to lock the weights when transfer learning
    # Set training=False to avoid updates to non-trainable BatchNorm parameters as described at
    # https://www.tensorflow.org/guide/keras/transfer_learning#build_a_model)
    base_model = _base_model(input_shape, weights=weights, trainable=base_trainable or weights is None)
    
    embedding = base_model(image, training=(None if weights is None else False))
    normalized_embedding = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embedding)

    projection = normalized_embedding if normalize_embedding else embedding
    if len(projection_size) > 0:
        if stop_gradient_before_projection:
            projection = tf.stop_gradient(projection)
        
        # Enable MLP (like supcon or SimCLR) where intermediate layers ReLU
        for dim in projection_size[:-1]:
            projection = layers.Dense(
                dim,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                use_bias=False,
                activation="relu",
            )(projection)
            projection = layers.BatchNormalization()(projection)
        
        # Last layer in projection
        projection = layers.Dense(projection_size[-1], use_bias=False, activation=None)(projection)
        if batch_normalize_projection:
            # SimCLR looks to batch normalize the last layer, while supcon does not
            projection = layers.BatchNormalization()(projection)
        if normalize_projection:
            projection = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(projection)

    return tf.keras.Model(inputs=image, outputs=[embedding, normalized_embedding, projection], name="encoder")


class GenotypingModel:    
    def summary(self):
        self._model.summary()

    def save(self, model_path: str):
        # We only save the encoder weights
        encoder = self._model.get_layer("encoder")
        encoder.save_weights(model_path)

    def _fit(self, cfg, training_dataset, validation_dataset=None):
        checkpoint_filepath = os.path.join(tempfile.gettempdir(), "checkpoints")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=True, monitor="loss", mode="min", save_best_only=True
        )
        #early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)

        callbacks=[checkpoint_callback] #,early_stopping]
    
        if cfg.training.log_dir:
            log_dir = os.path.join(cfg.training.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            logging.info("Logging TensorBoard data to: %s", log_dir)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False, histogram_freq=1, profile_batch=(10,15))
            callbacks.append(tensorboard_callback)

        self._model.fit(
            training_dataset,
            validation_data=validation_dataset,
            epochs=cfg.training.epochs,
            callbacks=callbacks,
        )
    
        # Reload best checkpoint
        self._model.load_weights(checkpoint_filepath)

class SupervisedBaselineModel(GenotypingModel):
    def __init__(self, image_shape, replicates, model_path: str=None, **kwargs):
        self.image_shape = image_shape
        self._model = self._create_model(image_shape, model_path=model_path, **kwargs)
        if model_path:
            self._model.load_weights(model_path)

    def save(self, model_path: str):
        self._model.save_weights(model_path)

    def _create_model(self, image_shape, model_path: str=None, **kwargs):
        encoder = _contrastive_encoder(
            image_shape,
            normalize_embedding=False,
            projection_size=[],
        )
         
        query = layers.Input(image_shape, name="query")
        query_embeddings, *_ = encoder(query)

        genotypes_logits = layers.Dropout(0.2)(query_embeddings)  # Regularize with dropout
        genotypes_logits = layers.Dense(3, name="genotypes_logits")(genotypes_logits)
        
        # Convert distance to probability
        genotypes = layers.Softmax(name="genotypes")(genotypes_logits)

        return tf.keras.Model(inputs=query, outputs=[genotypes, genotypes_logits])            

    def _train_input(self, cfg, dataset):
        assert len(self.image_shape) == 3
        
        def _variant_to_training(features, original_label):
            queries = tf.image.convert_image_dtype(features["image"], dtype=tf.float32)
    
            return ({ 
                "query": tf.image.convert_image_dtype(features["image"], dtype=tf.float32),
            }, {
                "genotypes_logits": original_label,
                "genotypes": original_label,
            })
        
        # Interleave SVs into the batch
        return (
            dataset
            .shuffle(cfg.training.shuffle, reshuffle_each_iteration=True)
            .map(_variant_to_training)
            .batch(cfg.training.variants_per_batch)
        )

    def fit(self, cfg, training_dataset, validation_dataset=None):
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.training.learning_rate)
        # TODO: Two stage learning for transfer learning?
        self._model.compile(
            optimizer=optimizer,
            loss={ "genotypes_logits": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) },
            metrics={ "genotypes": ["sparse_categorical_accuracy"] },
        )
                
        self._fit(cfg, self._train_input(cfg, training_dataset))

    def _test_input(self, cfg, dataset, batch_size):
        def _variant_to_test(features, original_label):
            return ({
                "query": tf.image.convert_image_dtype(features["image"], dtype=tf.float32),
            }, {
                "genotypes_logits": original_label,
                "genotypes": original_label,
            })

        return dataset.map(_variant_to_test).batch(batch_size)

    def predict(self, cfg, dataset, batch_size=1):
        return self._model.predict(self._test_input(cfg, dataset, batch_size=batch_size))


class SimulatedEmbeddingsModel(GenotypingModel):
    def __init__(self, image_shape, replicates, model_path: str=None, **kwargs):
        self.image_shape = image_shape
        self._model = self._create_model(image_shape, model_path=model_path, **kwargs)

    def _create_model(self, image_shape, model_path: str=None, **kwargs):
        assert len(image_shape) == 3, "Model only supports single images"
        
        # In this context, we only care about the projection output
        encoder =_contrastive_encoder(image_shape, weights=None, base_trainable=True, normalize_embedding=False, projection_size=[512], batch_normalize_projection=True, normalize_projection=True)
        _, _, projection = encoder.output
        encoder = tf.keras.Model(encoder.input, projection, name="encoder")
        
        if model_path:
            encoder.load_weights(model_path)
        
        support = layers.Input((3,) + image_shape, name="support")
        support_embeddings = layers.TimeDistributed(encoder, name="support_embeddings")(support)
        
        query = layers.Input(image_shape, name="query")
        query_embeddings = encoder(query)

        distances = layers.Lambda(_query_distances, name="distances")([query_embeddings, support_embeddings])
       
        # Convert distance to probability
        genotypes = layers.Lambda(lambda x: tf.nn.softmax(-x, axis=-1), name="genotypes")(distances) 

        return tf.keras.Model(inputs=[query, support], outputs=[genotypes, distances])


    def _train_input(self, cfg, dataset): 
        assert len(self.image_shape) == 3

        # Split the first replicate out as though it was three different queries...
        def _variant_to_training(features, original_label):
            support_shape = tf.shape(features["sim/images"])
            support_replicates = support_shape[1] - 1
            tf.debugging.assert_greater(support_replicates, 0)
            
            # Tile the queries so genotypes are interleaved
            queries = tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32)
            queries = tf.tile(queries, (support_replicates, 1, 1, 1))
            
            support = tf.image.convert_image_dtype(features["sim/images"][:,1:], dtype=tf.float32)
            support = tf.repeat(tf.transpose(support, perm=[1, 0, 2, 3, 4]), repeats=3, axis=0)

            return tf.data.Dataset.from_tensor_slices(({ 
                "query": queries,
                "support": support,
            }, {
                "distances": tf.tile(tf.one_hot([0, 1, 2], depth=3, dtype=tf.float32), (support_replicates, 1)),
                "genotypes": tf.tile(tf.constant([0, 1, 2]), (support_replicates,)),
            }))
        

        # Interleave SVs into the batch (in blocks of all three genotypes)
        return (dataset.shuffle(cfg.training.shuffle, reshuffle_each_iteration=True)
            .interleave(_variant_to_training, cycle_length=cfg.training.variants_per_batch, block_length=3)
            .batch(3*cfg.training.variants_per_batch)
            .prefetch(tf.data.AUTOTUNE)
        )
   

    def fit(self, cfg, training_dataset, validation_dataset=None):
        def _loss_wrapper(y_true, y_pred):
            # "Flatten" the three possible genotypes
            y_true = tf.dtypes.cast(tf.reshape(y_true, (-1,)), y_pred.dtype)
            y_pred = tf.reshape(y_pred, (-1,))
            weights = y_true * 1.5 + (1.0 - y_true) * 0.75  # There are always 2x more incorrect genotype pairs
            return tfa.losses.contrastive_loss(y_true, y_pred) * weights

        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.training.learning_rate),
            loss={ "distances": _loss_wrapper },
            metrics={ "genotypes": ["sparse_categorical_accuracy"] },
        )
        self._fit(cfg, self._train_input(cfg, training_dataset))


    def _test_input(self, cfg, dataset, batch_size):
        def _variant_to_test(features, original_label):
            one_hot_label = original_label if original_label is not None else -1
            return ({
                "query": tf.image.convert_image_dtype(features["image"], dtype=tf.float32),
                "support": tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32),
            }, {
                "distances": tf.one_hot(one_hot_label, depth=3, dtype=tf.float32),
                "genotypes": original_label,
            })

        return dataset.map(_variant_to_test).batch(batch_size)


    def predict(self, cfg, dataset, batch_size=1):
        return self._model.predict(self._test_input(cfg, dataset, batch_size=batch_size))


class JointEmbeddingsModel(SimulatedEmbeddingsModel):
    def __init__(self, image_shape, replicates, model_path: str=None, **kwargs):
        super().__init__(image_shape, replicates, model_path=model_path, **kwargs)

    
    def _train_input(self, cfg, dataset):
        file_descriptor_set = descriptor_pb2.FileDescriptorSet()
        npsv2_pb2.DESCRIPTOR.CopyToProto(file_descriptor_set.file.add())
        descriptor_source = b'bytes://' + file_descriptor_set.SerializeToString()
                
        def _variant_to_training(features, original_label):
            support_shape = tf.shape(features["sim/images"])
            support_replicates = support_shape[1]
            tf.debugging.assert_greater(support_replicates, 0)
            
            # TODO: Actually obtain reference size (since that determines the image size)
            _, variant_size = tf.io.decode_proto(
                features["variant/encoded"],
                "npsv2.StructuralVariant",
                ["svlen"],
                [tf.int64],
                descriptor_source=descriptor_source
            )
            variant_size = tf.squeeze(variant_size)
            
            queries = tf.image.convert_image_dtype(features["image"], dtype=tf.float32)
            queries = tf.repeat(tf.expand_dims(queries, axis=0), repeats=support_replicates, axis=0)

            # Swap AC and replicates dimensions, so we can pass triples of all genotypes to the model
            support = tf.image.convert_image_dtype(features["sim/images"], dtype=tf.float32)
            support = tf.transpose(support, perm=(1, 0) + tuple(range(2, 2+len(self.image_shape))))

            return tf.data.Dataset.from_tensor_slices(({ 
                "query": queries,
                "support": support,
                "variant_size": tf.repeat(variant_size, support_replicates),
            }, {
                "distances": tf.tile(tf.one_hot([original_label], depth=3, dtype=tf.float32), (support_replicates, 1)),
                "genotypes": tf.repeat(original_label, support_replicates),
            }))
        
        # Interleave SVs into the batch
        return (
            dataset
            .filter(lambda _, original_label: original_label is not None)
            .shuffle(cfg.training.shuffle, reshuffle_each_iteration=True)
            .interleave(_variant_to_training, cycle_length=cfg.training.variants_per_batch, num_parallel_calls=cfg.threads)
            #.filter(lambda inputs, _: inputs["variant_size"] > -300)
            .batch(cfg.training.variants_per_batch)
            .prefetch(tf.data.experimental.AUTOTUNE)  # Newer versions: tf.data.AUTOTUNE
        )

    def _test_input(self, cfg, dataset, batch_size):
        def _variant_to_test(features, original_label):
            support_shape = tf.shape(features["sim/images"])
            support_replicates = support_shape[1]
            tf.debugging.assert_greater(support_replicates, 0)

            queries = tf.image.convert_image_dtype(features["image"], dtype=tf.float32)
            queries = tf.repeat(tf.expand_dims(queries, axis=0), repeats=support_replicates, axis=0)

            # Swap AC and replicates dimensions, so we can pass triples of all genotypes to the model
            support = tf.image.convert_image_dtype(features["sim/images"], dtype=tf.float32)
            support = tf.transpose(support, perm=(1, 0) + tuple(range(2, 2+len(self.image_shape))))
            
            one_hot_label = original_label if original_label is not None else -1
        
            return tf.data.Dataset.from_tensor_slices(({ 
                "query": queries,
                "support": support,
            }, None))

            # return tf.data.Dataset.from_tensor_slices(({ 
            #     "query": queries,
            #     "support": support,
            # }, {
            #     "distances": tf.tile(tf.one_hot([one_hot_label], depth=3, dtype=tf.float32), (support_replicates, 1)),
            #     "genotypes": tf.repeat(original_label, support_replicates),
            # }))

        return dataset.flat_map(_variant_to_test).batch(batch_size)

class EncoderWrapperLayer(layers.Layer):
      def __init__(self, encoder):
        super(EncoderWrapperLayer, self).__init__()
        self.encoder = encoder

      def call(self, inputs):
        return self.encoder(inputs)

      def compute_output_shape(self, input_shape):
        return self.encoder.compute_output_shape(input_shape)

def _time_distributed_encoder(encoder, name=None):
    # This is a hack to use TimeDistributed with multiple outputs
    # https://github.com/keras-team/keras/issues/6449#issuecomment-298255231
    inputs = layers.Input(encoder.input_shape)
    outputs = []
    for output in encoder.output:
        outputs.append(layers.TimeDistributed(tf.keras.Model(encoder.input, output))(inputs))
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


class ProjectionJointEmbeddingsModel(JointEmbeddingsModel):
    def __init__(self, image_shape, replicates, model_path: str=None, **kwargs):
        super().__init__(image_shape, replicates, **kwargs)        

    def _create_model(self, image_shape, model_path: str=None, **kwargs):
        encoder = _contrastive_encoder(
            image_shape,
            normalize_embedding=True,
            stop_gradient_before_projection=False,
            projection_size=[128],
            normalize_projection=True,
            batch_normalize_projection=False
        )
        if model_path:
            encoder.load_weights(model_path)
        
        # We seem to need a wrapper to use TimeDistributed with multi-output models
        support = layers.Input((3,) + image_shape, name="support")
        _, support_embeddings, support_projections = _time_distributed_encoder(encoder, name="support_embeddings")(support)
       
        query = layers.Input(image_shape, name="query")
        _, query_embeddings, query_projections = encoder(query)

        embedding_distances = layers.Lambda(_query_distances, name="embedding_distances")([query_embeddings, support_embeddings])
        projection_distances = layers.Lambda(_query_distances, name="projection_distances")([query_projections, support_projections])

        # Convert distance to probability
        genotypes = layers.Lambda(lambda x: tf.nn.softmax(-x, axis=-1), name="genotypes")(embedding_distances) 

        return tf.keras.Model(inputs=[query, support], outputs=[genotypes, embedding_distances, projection_distances])        

    # TODO: Move input functions to mixins?
    def _train_input(self, cfg, dataset):
        assert len(self.image_shape) == 3
        
        def _variant_to_training(features, original_label):
            support_shape = tf.shape(features["sim/images"])
            support_replicates = support_shape[1]
            tf.debugging.assert_greater(support_replicates, 0)

            queries = tf.image.convert_image_dtype(features["image"], dtype=tf.float32)
            queries = tf.repeat(tf.expand_dims(queries, axis=0), repeats=support_replicates, axis=0)

            support = tf.image.convert_image_dtype(features["sim/images"], dtype=tf.float32)
            support = tf.transpose(support, perm=[1, 0, 2, 3, 4])

            return tf.data.Dataset.from_tensor_slices(({ 
                "query": queries,
                "support": support,
            }, {
                "projection_distances": tf.tile(tf.one_hot([original_label], depth=3, dtype=tf.float32), (support_replicates, 1)),
                "genotypes": tf.repeat(original_label, support_replicates),
            }))
        
        # Interleave SVs into the batch
        return (
            dataset.shuffle(cfg.training.shuffle, reshuffle_each_iteration=True)
            .interleave(_variant_to_training, cycle_length=cfg.training.variants_per_batch)
            .batch(cfg.training.variants_per_batch)
        )

    def fit(self, cfg, training_dataset, validation_dataset=None):
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.training.learning_rate)
        
        def _loss_wrapper(y_true, y_pred):
            # "Flatten" the three possible genotypes
            y_true = tf.dtypes.cast(tf.reshape(y_true, (-1,)), y_pred.dtype)
            y_pred = tf.reshape(y_pred, (-1,))
            return tfa.losses.contrastive_loss(y_true, y_pred)

            #weights = y_true * 1.5 + (1.0 - y_true) * 0.75  # There are always 2x more incorrect genotype pairs
            #return tfa.losses.contrastive_loss(y_true, y_pred) * weights

        self._model.compile(
            optimizer=optimizer,
            loss={ "projection_distances": _loss_wrapper },
            metrics={ "genotypes": ["sparse_categorical_accuracy"] },
        )
                
        self._fit(cfg, self._train_input(cfg, training_dataset))


class BreakpointJointEmbeddingsModel(JointEmbeddingsModel):
    def __init__(self, image_shape, replicates, model_path: str=None, **kwargs):
        assert len(image_shape) == 4 and image_shape[0] == 2, "Model assumes two images per variant"
        super().__init__(image_shape, replicates, **kwargs)

    def _create_model(self, image_shape, model_path: str=None, **kwargs):
        self._encoder = _contrastive_encoder(
            image_shape[1:],
            normalize_embedding=False,
            stop_gradient_before_projection=False,
            projection_size=[512],
            batch_normalize_projection=True,
            normalize_projection=False,
        )
        if model_path:
            self._encoder.load_weights(model_path)
        
        _, _, projection_shape = self._encoder.output_shape

        
        support = layers.Input((3,) + image_shape, name="support")
        
        # We seem to need a wrapper to use TimeDistributed with multi-output models. TimeDistributed is limited to
        # 5-D so reshape AC (size 3) and breakpoint dimensions (size 2) into a single dimension of size 6
        support_images = layers.Reshape((6,) + image_shape[1:])(support) 
        _, _, support_projections = _time_distributed_encoder(self._encoder, name="support_embeddings")(support_images)
        support_projections = layers.Reshape((3, 2 * projection_shape[-1]))(support_projections)
        
        # Normalize concatenated projection (both images concatenated)
        support_projections = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(support_projections)

        # Create other branch of Siamese network for query images
        query = layers.Input(image_shape, name="query")
        _, _, query_projections = _time_distributed_encoder(self._encoder, name="query_embeddings")(query)
        query_projections = layers.Reshape((2 * projection_shape[-1],))(query_projections)
        query_projections = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(query_projections)

        # Compute distances between query and support
        projection_distances = layers.Lambda(_query_distances, name="distances")([query_projections, support_projections])

        # Convert distance to probability
        genotypes = layers.Lambda(lambda x: tf.nn.softmax(-x, axis=-1), name="genotypes")(projection_distances) 

        return tf.keras.Model(inputs=[query, support], outputs=[genotypes, projection_distances])        

    def save(self, model_path: str):
        # We only save the encoder weights
        self._encoder.save_weights(model_path)

