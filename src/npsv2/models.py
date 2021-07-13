import datetime, logging, os, tempfile
from omegaconf import OmegaConf
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from deeplab2.model.encoder import axial_resnet_instances

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

def _get_backbone(name, input_shape):
    backbone = axial_resnet_instances.get_model("resnet50", backbone_layer_multiplier=1, bn_layer=tf.keras.layers.BatchNormalization, conv_kernel_weight_decay=0.0001)
     
    image = layers.Input(input_shape) 
    output = backbone(image)
    embedding = layers.GlobalAveragePooling2D()(output["res5"])
    model = tf.keras.Model(inputs=image, outputs=embedding)
    return model


class AxialWrapperLayer(layers.Layer):
    def __init__(self, encoder):
        super(AxialWrapperLayer, self).__init__()
        self.encoder = encoder

    def call(self, inputs):
        return self.encoder(inputs)

    def compute_output_shape(self, input_shape):
        return (None, 2048)

    # base_model = tf.keras.applications.InceptionV3(include_top=False, weights=None, input_shape=input_shape, pooling="avg")
   
    # # imagenet weights require a 3-channel input image. To enable us to use more channels, we need to construct a ../
    # src_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape[:-1] + (3,), pooling="avg")
     
    # # Only the first convolution layer needs to have different weights
    # for i, (src_layer, dst_layer) in enumerate(zip(src_model.layers, base_model.layers)):
    #     if i == 1:
    #         kernels, *other_weights = src_layer.get_weights()
            
    #         # Replicate mean of exAssumes "channels_last"
    #         new_kernels = tf.repeat(tf.reduce_mean(kernels, axis=-2, keepdims=True) * (3/input_shape[-1]), input_shape[-1], axis=-2)
    #         dst_layer.set_weights([new_kernels] + other_weights)
    #     else:
    #         dst_layer.set_weights(src_layer.get_weights())


    # #base_model = tf.keras.applications.InceptionV3(include_top=False, weights=None, input_shape=input_shape, pooling="avg")
    # base_model.trainable = True
    
    # encoder = tf.keras.models.Sequential([
    #     layers.InputLayer(input_shape),
    #     #layers.Conv2D(3, (1,1), activation='tanh'),  # Make multi-channel input compatible with Inception (input [-1, 1])
    #     base_model,
    #     layers.Dense(512),
    #     layers.BatchNormalization(),
    # ], name="encoder")
    # if normalize:
    #     encoder.add(layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))) # L2 normalize embeddings
    # return encoder

    # image = layers.Input(input_shape)
    # x = layers.Conv2D(3, (1,1), activation='tanh')(image)
    # x = base_model(x, training=False)
    # x = layers.Dense(512)(x)
    # x = layers.BatchNormalization()(x)
    # if normalize:
    #     x = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    # return tf.keras.Model(inputs=image, outputs=x, name="encoder")
    

   

# Adapted from:
# https://github.com/google-research/google-research/blob/e2308c7593eda306daab40db07930a2d5132255b/supcon/models.py#L26
def _contrastive_encoder(input_shape, weights=None, base_trainable=True, normalize_embedding=True, stop_gradient_before_projection=False, projection_size=[128], batch_normalize_projection=False, normalize_projection=True):
    assert tf.keras.backend.image_data_format() == "channels_last"

    image = layers.Input(input_shape) 
    
    # Set trainable to False to lock the weights when transfer learning
    # Set training=False to avoid updates to non-trainable BatchNorm parameters as described at
    # https://www.tensorflow.org/guide/keras/transfer_learning#build_a_model)
    base_model = _base_model(input_shape, weights=weights, trainable=base_trainable or weights is None)
    #base_model = AxialWrapperLayer(_get_backbone("test", input_shape))
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

        #  # Train the additional layers
        # layer_phase_cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(["training.epochs=20"]))       
        # self._model.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=layer_phase_cfg.training.learning_rate),
        #     loss={ "distances": _loss_wrapper },
        #     metrics={ "genotypes": ["sparse_categorical_accuracy"] },
        # )
        # self._fit(layer_phase_cfg, self._train_input(layer_phase_cfg, training_dataset))
       

        # # Fine tune the model by unlock all the layers
        # finetune_phase_cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(["training.epochs=20"]))
        # self._model.trainable = True
        # self._model.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004),
        #     loss={ "distances": _loss_wrapper },
        #     metrics={ "genotypes": ["sparse_categorical_accuracy"] },
        # )
        # self._fit(finetune_phase_cfg, self._train_input(finetune_phase_cfg, training_dataset))

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
        def _variant_to_training(features, original_label):
            support_shape = tf.shape(features["sim/images"])
            support_replicates = support_shape[1]
            tf.debugging.assert_greater(support_replicates, 0)

            queries = tf.image.convert_image_dtype(features["image"], dtype=tf.float32)
            queries = tf.repeat(tf.expand_dims(queries, axis=0), repeats=support_replicates, axis=0)

            # Swap AC and replicates dimensions, so we can pass triples of all genotypes to the model
            support = tf.image.convert_image_dtype(features["sim/images"], dtype=tf.float32)
            support = tf.transpose(support, perm=(1, 0) + tuple(range(2, 2+len(self.image_shape))))

            return tf.data.Dataset.from_tensor_slices(({ 
                "query": queries,
                "support": support,
            }, {
                "distances": tf.tile(tf.one_hot([original_label], depth=3, dtype=tf.float32), (support_replicates, 1)),
                "genotypes": tf.repeat(original_label, support_replicates),
            }))
        
        # Interleave SVs into the batch
        return (
            dataset.filter(lambda _, original_label: original_label is not None)
            .shuffle(cfg.training.shuffle, reshuffle_each_iteration=True)
            .interleave(self._variant_to_training, cycle_length=cfg.training.variants_per_batch, num_parallel_calls=cfg.threads)
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


# def _variant_to_allsim_training_triples(features, original_label):
#     # TODO: Make this a function of the number of replicates
#     query_images = tf.tile(
#         tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32),
#         [4, 1, 1, 1]
#     )
#     support_images = tf.stack([
#         tf.image.convert_image_dtype(features["sim/images"][:,1], dtype=tf.float32),
#         tf.image.convert_image_dtype(features["sim/images"][:,1], dtype=tf.float32),
#         tf.image.convert_image_dtype(features["sim/images"][:,1], dtype=tf.float32),
#         tf.image.convert_image_dtype(features["sim/images"][:,2], dtype=tf.float32),
#         tf.image.convert_image_dtype(features["sim/images"][:,2], dtype=tf.float32),
#         tf.image.convert_image_dtype(features["sim/images"][:,2], dtype=tf.float32),
#         tf.image.convert_image_dtype(features["sim/images"][:,3], dtype=tf.float32),
#         tf.image.convert_image_dtype(features["sim/images"][:,3], dtype=tf.float32),
#         tf.image.convert_image_dtype(features["sim/images"][:,3], dtype=tf.float32),
#         tf.image.convert_image_dtype(features["sim/images"][:,4], dtype=tf.float32),
#         tf.image.convert_image_dtype(features["sim/images"][:,4], dtype=tf.float32),
#         tf.image.convert_image_dtype(features["sim/images"][:,4], dtype=tf.float32),
#     ])
#     image_tensors = {
#         "query": query_images,
#         "support": support_images,
#     }
#     return tf.data.Dataset.from_tensor_slices((image_tensors, tf.constant([0, 1, 2]*4, dtype=tf.int64)))

# def _variant_to_real_training_triples(features, original_label):
#     # TODO: Make this a function of the number of replicates
#     query_images = tf.stack([tf.image.convert_image_dtype(features["image"], dtype=tf.float32)]*4)
#     support_images = tf.stack([
#         tf.image.convert_image_dtype(features["sim/images"][:,1], dtype=tf.float32),
#         tf.image.convert_image_dtype(features["sim/images"][:,2], dtype=tf.float32),
#         tf.image.convert_image_dtype(features["sim/images"][:,3], dtype=tf.float32),
#         tf.image.convert_image_dtype(features["sim/images"][:,4], dtype=tf.float32),
#     ])

#     image_tensors = {
#         "query": query_images,
#         "support": support_images,
#     }

#     return tf.data.Dataset.from_tensor_slices((image_tensors, tf.repeat(original_label, 4)))
 

# def _variant_to_test_triples(features, original_label):
#     query_images = tf.image.convert_image_dtype(features["image"], dtype=tf.float32)
#     support_images = tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32)
   
#     image_tensors = {
#         "query": query_images,
#         "support": support_images,
#     }
#     return image_tensors, original_label, features["variant/encoded"]

# class SiameseGenotyper:
#     def __init__(self, input_shape, replicates):
#         self.input_shape = input_shape
#         self.replicates = replicates
    
#     def train_input(self, filenames, shuffle=1000, batch_size=24):
#         example_dataset = load_example_dataset(filenames, with_label=True, with_simulations=True)

#         example_dataset = example_dataset.shuffle(shuffle, reshuffle_each_iteration=True)
#         example_dataset = example_dataset.flat_map(_variant_to_allsim_training_triples)
#         #example_dataset = example_dataset.flat_map(_variant_to_real_training_triples)
#         example_dataset = example_dataset.batch(batch_size)
        
#         return example_dataset

#     def test_input(self, filenames):
#         example_dataset = load_example_dataset(filenames, with_label=True, with_simulations=True)
#         return example_dataset.map(_variant_to_test_triples).batch(1)

#     def validation_input(self, filenames, batch_size=24):
#         example_dataset = load_example_dataset(filenames, with_label=True, with_simulations=True)

#         # Strip off variant description and batch the same way as the training data
#         return example_dataset.map(_variant_to_test_triples).map(lambda image_tensors, original_label, _: (image_tensors, original_label)).batch(batch_size)
#         #return None

#     def model_fn(self, params=None, model_path:str=None):
#         encoder = _inceptionv3_encoder(self.input_shape)
        
#         query = layers.Input(self.input_shape, name="query")
#         query_embeddings = encoder(query)
        
#         support = layers.Input((3,) + self.input_shape, name="support")
#         support_embeddings = layers.TimeDistributed(encoder)(support)
        
#         def _variant_distances(tensors):
#             query, support = tensors
#             return tf.squeeze(tf.map_fn(_cdist, (tf.expand_dims(query, axis=1), support), dtype=tf.dtypes.float32), axis=1)

#         distances = layers.Lambda(_variant_distances, name="distances")([query_embeddings, support_embeddings])
#         output = layers.Lambda(lambda x: tf.nn.softmax(-x, axis=1), name="genotypes")(distances) # Convert distance to probability

#         model = tf.keras.Model(inputs=[query, support], outputs=[output, distances], name="SiameseGenotyper")
#         if model_path:
#             encoder.load_weights(model_path)

#         if params:
#             optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
#             model.compile(
#                 optimizer=optimizer,
#                 loss={"genotypes": "sparse_categorical_crossentropy"},
#                 metrics={"genotypes": ["sparse_categorical_accuracy"]},
#             )

#         return model

#     def save_model(self, model, model_path: str):
#         # We only save the encoder weights
#         encoder = model.get_layer("encoder")
#         encoder.save_weights(model_path)

# class GenotypingModelConfig(Config):
#     tempdir: str = tempfile.gettempdir()
#     log_dir: str = None
#     learning_rate: float = 0.004
#     epochs: int = 20
#     threads: int = 1

# class GenotypingModel:    
#     def summary(self):
#         self._model.summary()

#     def save(self, model_path: str):
#         # We only save the encoder weights
#         encoder = self._model.get_layer("encoder")
#         encoder.save_weights(model_path)

#     def _fit(self, config, training_dataset, validation_dataset=None, **kwargs):
#         checkpoint_filepath = os.path.join(config.tempdir, "checkpoint")
#         checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#             filepath=checkpoint_filepath, save_weights_only=True, monitor="loss", mode="min", save_best_only=True
#         )
#         #early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)

#         callbacks=[checkpoint_callback] #,early_stopping]
    
#         if config.log_dir:
#             log_dir = os.path.join(config.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#             logging.info("Logging TensorBoard data to: %s", log_dir)
#             tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#             callbacks.append(tensorboard_callback)

#         self._model.fit(
#             training_dataset,
#             validation_data=validation_dataset,
#             epochs=config.epochs,
#             callbacks=callbacks,
#         )
    
#         # Reload best checkpoint
#         self._model.load_weights(checkpoint_filepath)


# class TripletModelConfig(GenotypingModelConfig):
#     variants_per_batch: int = 8
#     shuffle: int = 1000


# class TripletModel(GenotypingModel):
#     def __init__(self, image_shape, replicates, model_path: str=None, **kwargs):
#         self.config = merge_config(TripletModelConfig(), kwargs)
        
#         self.image_shape = image_shape
#         self.replicates = replicates

#         encoder = _inceptionv3_encoder(image_shape, normalize=True)       
#         if model_path:
#             encoder.load_weights(model_path)
        
#         self._model = TripletModel._model_from_encoder(encoder, image_shape, replicates)

#     @classmethod
#     def _model_from_encoder(cls, encoder, image_shape, replicates):
#         embedding_shape = encoder.output_shape
        
#         support = layers.Input((3, replicates) + image_shape, name="support")
#         support_embeddings = layers.Reshape((3 * replicates,) + image_shape)(support)
#         support_embeddings = layers.TimeDistributed(encoder, name="support_embeddings")(support_embeddings)
       
#         query = layers.Input(image_shape, name="query")
#         query_embeddings = encoder(query)
        
#         def _query_distances(tensors):
#             query_embeddings, support_batch = tensors

#             # Make query embeddings have shape (batch, 1, ...) so we can map across batch to compute
#             # distances between query example and all support examples
#             support_distances = tf.map_fn(_cdist, (tf.expand_dims(query_embeddings, axis=1), support_batch), dtype=tf.dtypes.float32)
#             support_distances = tf.squeeze(support_distances, axis=1)  # Remove unit dimension for query

#             # An alternate approach could use the genotype "cluster" means instead of cluster minimum
#             cluster_distances = tf.math.reduce_min(tf.reshape(support_distances, (-1, 3, replicates)), axis=-1)
#             return cluster_distances

#         distances = layers.Lambda(_query_distances, name="distances")([query_embeddings, support_embeddings])
        
#         # Convert distance to probability
#         genotypes = layers.Lambda(lambda x: tf.nn.softmax(-x, axis=-1), name="genotypes")(distances) 

#         return tf.keras.Model(inputs=[query, support], outputs=[genotypes, distances, support_embeddings])

    
#     def _train_input(self, config, dataset):
#         def _variant_to_training(index, element):
#             features, original_label = element
#             triplet_labels = range(3*index, 3*(index + 1))
#             return ({ 
#                 "query": tf.image.convert_image_dtype(features["image"], dtype=tf.float32),
#                 "support": tf.image.convert_image_dtype(features["sim/images"], dtype=tf.float32),
#             }, {
#                 "support_embeddings": tf.repeat(triplet_labels, repeats=self.replicates),
#                 "genotypes": original_label,
#             })
        
#         return dataset.shuffle(config.shuffle, reshuffle_each_iteration=True).enumerate().map(_variant_to_training).batch(config.variants_per_batch)
        

#     def fit(self, training_dataset, validation_dataset=None, **kwargs):
#         config = merge_config(self.config, kwargs)

#         optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        
#         def _loss_wrapper(y_true, y_pred):
#             # "Flatten" batch size (assuming unique label for each variant)
#             embedding_shape = tf.shape(y_pred)
#             y_true = tf.reshape(y_true, (-1,))
#             y_pred = tf.reshape(y_pred, (-1, embedding_shape[-1]))
#             return tfa.losses.triplet_semihard_loss(y_true, y_pred)

#         self._model.compile(
#             optimizer=optimizer,
#             #loss={ "support_embeddings": _loss_wrapper, "genotypes": "sparse_categorical_crossentropy" },
#             loss={ "genotypes": "sparse_categorical_crossentropy" },
#             metrics={ "genotypes": ["sparse_categorical_accuracy"] },
#         )
                
#         self._model.fit(
#             self._train_input(config, training_dataset),
#             epochs=config.epochs,
#         )

#     def _test_input(self, dataset):
#         def _variant_to_test(features, original_label):
#             return {
#                 "query": tf.image.convert_image_dtype(features["image"], dtype=tf.float32),
#                 "support": tf.expand_dims(tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32), axis=1),
#             }, original_label

#         return dataset.map(_variant_to_test).batch(1)


#     def make_predict(self, **kwargs):
#         # Build a simplified version of the model that expects only one replicate
#         encoder = self._model.get_layer("encoder")
#         model = TripletModel._model_from_encoder(encoder, self.image_shape, replicates=1)
        
#         return lambda dataset: model.predict(self._test_input(dataset))

#     def predict(self, dataset, **kwargs):
#         predict_fn = self.make_predict(**kwargs)
#         return predict_fn(dataset)


# class JointEmbeddingsModelConfig(GenotypingModelConfig):
#     variants_per_batch: int = 8
#     shuffle: int = 1000


# class JointEmbeddingsModel(GenotypingModel):
#     def __init__(self, image_shape, replicates, model_path: str=None, **kwargs):
#         self.config = merge_config(JointEmbeddingsModelConfig(), kwargs)
        
#         self.image_shape = image_shape
#         self.replicates = replicates

#         encoder = _inceptionv3_encoder(image_shape, normalize=True)       
#         if model_path:
#             encoder.load_weights(model_path)
        
#         self._model = JointEmbeddingsModel._model_from_encoder(encoder, image_shape, replicates)
    
#     @classmethod
#     def _model_from_encoder(cls, encoder, image_shape, replicates):
#         embedding_shape = encoder.output_shape
        
#         support = layers.Input((3, replicates) + image_shape, name="support")
#         support_embeddings = layers.Reshape((3 * replicates,) + image_shape)(support)
#         support_embeddings = layers.TimeDistributed(encoder, name="support_embeddings")(support_embeddings)
       
#         query = layers.Input(image_shape, name="query")
#         query_embeddings = encoder(query)
        
#         def _query_distances(tensors):
#             query_embeddings, support_batch = tensors

#             # Make query embeddings have shape (batch, 1, ...) so we can map across batch to compute
#             # distances between query example and all support examples
#             support_distances = tf.map_fn(_cdist, (tf.expand_dims(query_embeddings, axis=1), support_batch), dtype=tf.dtypes.float32)
#             support_distances = tf.squeeze(support_distances, axis=1)  # Remove unit dimension for query

#             return support_distances

#         support_distances = layers.Lambda(_query_distances, name="support_distances")([query_embeddings, support_embeddings])
        
#         # An alternate approach could use the genotype "cluster" means instead of cluster minimum
#         distances = layers.Lambda(lambda x: tf.math.reduce_min(tf.reshape(x, (-1, 3, replicates)), axis=-1), name="distances")(support_distances)
        
#         # Convert distance to probability
#         genotypes = layers.Lambda(lambda x: tf.nn.softmax(-x, axis=-1), name="genotypes")(distances) 

#         return tf.keras.Model(inputs=[query, support], outputs=[genotypes, distances, support_distances])

#     def _train_input(self, config, dataset):
#         def _variant_to_training(features, original_label):
#             contrastive_labels = tf.one_hot(original_label, depth=3, dtype=tf.float32)

#             return ({ 
#                 "query": tf.image.convert_image_dtype(features["image"], dtype=tf.float32),
#                 "support": tf.image.convert_image_dtype(features["sim/images"], dtype=tf.float32),
#             }, {
#                 "support_distances": tf.repeat(contrastive_labels, repeats=self.replicates),
#                 "genotypes": original_label,
#             })
        
#         return dataset.shuffle(config.shuffle, reshuffle_each_iteration=True).map(_variant_to_training).batch(config.variants_per_batch)
        

#     def fit(self, training_dataset, validation_dataset=None, **kwargs):
#         config = merge_config(self.config, kwargs)

#         optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        
#         def _loss_wrapper(y_true, y_pred):
#             # "Flatten" batch size
#             y_true = tf.dtypes.cast(tf.reshape(y_true, (-1,)), y_pred.dtype)
#             y_pred = tf.reshape(y_pred, (-1,))
#             weights = y_true * 1.5 + (1.0 - y_true) * 0.75  # There are always 2x more incorrect genotype pairs
#             return tfa.losses.contrastive_loss(y_true, y_pred) * weights

#         self._model.compile(
#             optimizer=optimizer,
#             loss={ "support_distances": _loss_wrapper },
#             metrics={ "genotypes": ["sparse_categorical_accuracy"] },
#         )
                
#         self._model.fit(
#             self._train_input(config, training_dataset),
#             epochs=config.epochs,
#         )

#     def _test_input(self, config, dataset):
#         def _variant_to_test(features, original_label):
#             return {
#                 "query": tf.image.convert_image_dtype(features["image"], dtype=tf.float32),
#                 "support": tf.expand_dims(tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32), axis=1),
#             }, original_label

#         return dataset.map(_variant_to_test).batch(1)

#     def predict(self, dataset, **kwargs):
#         assert self.replicates == 1, "Predict assumes a single replicate"

#         config = merge_config(self.config, kwargs)
#         return self._model.predict(self._test_input(config, dataset))





# def _residual_block(embedding_shape, filters, kernel_size=2, dilation_rate=1, padding="causal", dropout_rate=0.2, name="residual_block"):
#     init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)

#     sequence = layers.Input(embedding_shape, name="sequence")
#     transform = tf.keras.models.Sequential([
#         layers.Input(embedding_shape),
#         layers.Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding, kernel_initializer=init),
#         layers.BatchNormalization(axis=-1),
# 		layers.Activation("relu"),
# 		layers.Dropout(rate=dropout_rate),
#         layers.Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding, kernel_initializer=init),
#         layers.BatchNormalization(axis=-1),
# 		layers.Activation("relu"),
# 		layers.Dropout(rate=dropout_rate),
#         # tfa.layers.WeightNormalization(layers.Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding, kernel_initializer=init, activation="relu")),
#         # layers.SpatialDropout1D(rate=dropout_rate),
#         # tfa.layers.WeightNormalization(layers.Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding, kernel_initializer=init, activation="relu")),
#         # layers.SpatialDropout1D(rate=dropout_rate),
#     ], name="residual_layers")(sequence)

      

#     matched_sequence = sequence
#     if filters != embedding_shape[-1]:
#         # Match the input and temporal block shapes
#         matched_sequence = layers.Conv1D(filters=filters, kernel_size=1, padding="same", kernel_initializer=init)(sequence)

#     residual = layers.Add()([matched_sequence, transform])
#     residual = layers.Activation("relu")(residual)
#     return tf.keras.Model(inputs=sequence, outputs=residual)


# def _resnet50v2_encoder(input_shape, normalize=False, dropout_rate=0.2, embedding_size=128):
#     assert tf.keras.backend.image_data_format() == "channels_last"

#     #base_model = tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_shape=input_shape, pooling="avg")
#     #base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=input_shape, pooling=None)
#     base_model = tf.keras.applications.ResNet50V2(include_top=False, weights="imagenet", input_shape=input_shape[:-1] + (3,), pooling="avg")
    
#     base_model.trainable = True
    
#     encoder = tf.keras.models.Sequential([
#         layers.Input(input_shape),
#         #layers.Lambda(lambda x: tf.keras.applications.resnet_v2.preprocess_input(x)),
#         #layers.Lambda(lambda x: tf.keras.applications.efficientnet.preprocess_input(x)),
#         layers.Conv2D(3, (1,1), activation='tanh'),  # Make multi-channel input compatible with ResNet (input [-1, 1])
#         base_model,
#         layers.BatchNormalization(),
#         layers.Dropout(rate=dropout_rate),
#         layers.Flatten(),
#         layers.Dense(embedding_size),
#         layers.BatchNormalization(),
#     ], name="image_encoder")
#     if normalize:
#         encoder.add(layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))) # L2 normalize embeddings
#     return encoder


# def _tcn_encoder(input_shape, tcn_filters=128):
#     windows = layers.Input((None,) + input_shape[-3:], name="windows")

#     window_encoder = _resnet50v2_encoder(input_shape=input_shape[-3:], normalize=False)
#     window_embeddings = layers.TimeDistributed(window_encoder, name="window_embeddings")(windows)
   
#     tcn_input_shape = (None, window_encoder.output_shape[-1])
#     tcn_output_shape = (None, tcn_filters)

#     # This should create a receptive field of 24 windows... (4-1)*8
#     tcn = tf.keras.models.Sequential([
#         _residual_block(tcn_input_shape, tcn_filters, dilation_rate=1, name="residual_block_1"),
#         _residual_block(tcn_output_shape, tcn_filters, dilation_rate=2, name="residual_block_2"),
#         _residual_block(tcn_output_shape, tcn_filters, dilation_rate=4, name="residual_block_4"),
#         _residual_block(tcn_output_shape, tcn_filters, dilation_rate=8, name="residual_block_8"),
#         _residual_block(tcn_output_shape, tcn_filters, dilation_rate=16, name="residual_block_16"),
#     ], name="tcn")

#     #tcn = _residual_block((None, window_embedding_shape[-1]), tcn_filters)
#     tcn_embeddings = tcn(window_embeddings)
#     tcn_embeddings = layers.Lambda(lambda tcn_seq: tcn_seq[:, -1, :])(tcn_embeddings) # Select last value in sequence

#     return tf.keras.Model(inputs=windows, outputs=tcn_embeddings, name="encoder")


# def _support_reshape(support):
#     # https://stackoverflow.com/a/54983612
#     support_shape = tf.shape(support)
#     distributed_input_shape = tf.concat(
#         [support_shape[:-6], tf.math.reduce_prod(support_shape[-6:-4], keepdims=True), support_shape[-4:]], 0,
#     )
#     return tf.reshape(support, distributed_input_shape)


# class WindowedJointEmbeddingsModelConfig(GenotypingModelConfig):
#     variants_per_batch: int = 1
#     shuffle: int = 1000

# class WindowedJointEmbeddingsModel(GenotypingModel):
#     def __init__(self, image_shape, replicates=1, model_path: str=None, **kwargs):
#         assert len(image_shape) == 3, "Expect 3-D image shape"
#         self.config = merge_config(WindowedJointEmbeddingsModelConfig(), kwargs)
        
#         self.image_shape = image_shape
#         self.replicates = replicates

#         encoder = _tcn_encoder(image_shape)
    
#         if model_path:
#             encoder.load_weights(model_path)
        
#         self._model = WindowedJointEmbeddingsModel._model_from_encoder(encoder, image_shape, replicates)
    
#     @classmethod
#     def _model_from_encoder(cls, encoder, image_shape, replicates):
#         support = layers.Input((3, None) + image_shape, name="support")
#         query = layers.Input((None,) + image_shape, name="query")

#         support_embeddings = layers.TimeDistributed(encoder, name="support_embeddings")(support)
#         query_embeddings = encoder(query)
        
#         def _query_distances(tensors):
#             query_embeddings, support_batch = tensors

#             # Make query embeddings have shape (batch, 1, ...) so we can map across batch to compute
#             # distances between query example and all support examples
#             support_distances = tf.map_fn(_cdist, (tf.expand_dims(query_embeddings, axis=1), support_batch), dtype=tf.dtypes.float32)
#             support_distances = tf.squeeze(support_distances, axis=1)  # Remove unit dimension for query

#             return support_distances

#         support_distances = layers.Lambda(_query_distances, name="distances")([query_embeddings, support_embeddings])
#         # An alternate approach could use the genotype "cluster" means instead of cluster minimum
#         #distances = layers.Lambda(lambda x: tf.math.reduce_min(tf.reshape(x, (-1, 3, replicates)), axis=-1), name="distances")(support_distances)
        
#         # Convert distance to probability
#         genotypes = layers.Lambda(lambda x: tf.nn.softmax(-x, axis=-1), name="genotypes")(support_distances) 
#         return tf.keras.Model(inputs=[query, support], outputs=[genotypes, support_distances])

#     def _train_input(self, config, dataset):
#         def _variant_to_training(features, original_label):
#             contrastive_labels = tf.one_hot(original_label, depth=3, dtype=tf.float32)
            
#             # At present is doesn't look like we can use the tf.data.experimental.bucket_by_sequence_length function here.
#             # At a minimum we would need to manually set the shape for the images (it seems to be lost in this function), but
#             # the padding seems to be limited to rank 5 and support has rank 6.
#             query_images = tf.image.convert_image_dtype(features["image"], dtype=tf.float32)
#             query_images = tf.repeat(tf.expand_dims(query_images, axis=0), self.replicates, axis=0)
#             # For some reason we need to set the shape after the above operations here...
#             query_images.set_shape((self.replicates, None,) + self.image_shape)

#             support_images = tf.image.convert_image_dtype(features["sim/images"], dtype=tf.float32)
#             support_images.set_shape((3, self.replicates, None) + self.image_shape)
#             support_images = tf.transpose(support_images, (1, 0, 2, 3, 4, 5))

#             return tf.data.Dataset.from_tensor_slices(({ 
#                 "query": query_images,
#                 "support": support_images,
#             }, tf.repeat(original_label, self.replicates)))
        
#         #return dataset.flat_map(_variant_to_training).batch(self.replicates)  

#         def _num_windows(x, y=None):
#             query = x["query"]
#             return tf.shape(query)[0]  # The number of windows 

#         buckets = list(range(5, 25))

#         bucketing = tf.data.experimental.bucket_by_sequence_length(
#             _num_windows, 
#             bucket_boundaries=buckets,
#             bucket_batch_sizes=([32]*5) + ([16]*10) + ([8]*6),  
#             drop_remainder=False,
#             pad_to_bucket_boundary=False,
#         )

#         return dataset.interleave(_variant_to_training, block_length=self.replicates, num_parallel_calls=config.threads, deterministic=False).shuffle(config.shuffle, reshuffle_each_iteration=True).apply(bucketing)
        
#     def _validation_input(self, config, dataset):
#         def _variant_to_training(features, original_label):
#             contrastive_labels = tf.one_hot(original_label, depth=3, dtype=tf.float32)

#             print(features["sim/images"].shape)
#             support_images = tf.tile(tf.image.convert_image_dtype(features["sim/images"][:,0:1], dtype=tf.float32), (1, self.replicates, 1, 1, 1, 1))
#             print(support_images.shape)

#             return ({ 
#                 "query": tf.image.convert_image_dtype(features["image"], dtype=tf.float32),
#                 "support": support_images,
#             }, {
#                 "support_distances": tf.repeat(contrastive_labels, repeats=self.replicates),
#                 "genotypes": original_label,
#             })
        
#         return dataset.map(_variant_to_training).batch(1)
 

#     def fit(self, training_dataset, validation_dataset=None, **kwargs):
#         config = merge_config(self.config, kwargs)

#         # for x, y in self._train_input(config, training_dataset):
#         #     print(x["query"].shape, x["support"].shape, y)
#         #assert False
#         optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        
#         def _loss_wrapper(y_true, y_pred):
#             # Convert sparse labels to binary match/non-match and "flatten" genotypes across the batch
#             y_true = tf.one_hot(tf.squeeze(y_true), depth=3, dtype=y_pred.dtype)
#             y_true = tf.reshape(y_true, (-1,))
#             y_pred = tf.reshape(y_pred, (-1,))
#             return tfa.losses.contrastive_loss(y_true, y_pred)
#             #weights = y_true * 1.5 + (1.0 - y_true) * 0.75  # There are always 2x more incorrect genotype pairs
#             #return tfa.losses.contrastive_loss(y_true, y_pred) * weights

#         self._model.compile(
#             optimizer=optimizer,
#             loss={ "distances": _loss_wrapper },
#             metrics={ "genotypes": ["sparse_categorical_accuracy"] },
#         )
                
#         self._fit(
#             config,
#             training_dataset=self._train_input(config, training_dataset),
#             validation_dataset=self._train_input(config, training_dataset), #validation_dataset,
#         )

#     def _test_input(self, config, dataset):
#         def _variant_to_test(features, original_label):
#             return {
#                 "query": tf.image.convert_image_dtype(features["image"], dtype=tf.float32),
#                 "support": tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32),
#             }, original_label

#         return dataset.map(_variant_to_test).batch(1)


#     def predict(self, dataset, **kwargs):
#         assert self.replicates == 1, "Predict assumes a single replicate"

#         config = merge_config(self.config, kwargs)
#         return self._model.predict(self._test_input(config, dataset))