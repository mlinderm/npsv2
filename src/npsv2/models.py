import datetime, logging, os, re, sys, tempfile, typing, warnings
import collections.abc
from omegaconf import OmegaConf
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from google.protobuf import descriptor_pb2

from .images import load_example_dataset, _extract_metadata_from_first_example, _features_variant
from . import npsv2_pb2
from .utilities.callbacks import NModelCheckpoint
from .utilities.sequence import as_scalar
from .utilities.losses import contrastive_loss

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
        # Suppress warnings about input shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # imagenet weights require a 3-channel input image. To enable us to use more channels, we construct a dummy model
            # with 3-channel image and copy those weights where possible into a network with the desired size
            base_model = tf.keras.applications.InceptionV3(include_top=False, weights=None, input_shape=input_shape, pooling="avg")

            # Only the first convolution layer needs to have different weights
            src_model = tf.keras.applications.InceptionV3(include_top=False, weights=weights, input_shape=input_shape[:-1] + (3,), pooling="avg")
        
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
        # Suppress warnings about input shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_model = tf.keras.applications.InceptionV3(include_top=False, weights=weights, input_shape=input_shape, pooling="avg")
            base_model.trainable = trainable
    
    return base_model



# Adapted from:
# https://github.com/google-research/google-research/blob/e2308c7593eda306daab40db07930a2d5132255b/supcon/models.py#L26
def _contrastive_encoder(input_shape, weights=None, base_trainable=True, normalize_embedding=True, stop_gradient_before_projection=False, projection_size=[128], batch_normalize_projection=False, normalize_projection=True, typed_projection=False, selected_type_indices=None):
    assert tf.keras.backend.image_data_format() == "channels_last"

    image = layers.Input(input_shape) 
    
    # Set trainable to False to lock the weights when transfer learning
    # Set training=False to avoid updates to non-trainable BatchNorm parameters as described at
    # https://www.tensorflow.org/guide/keras/transfer_learning#build_a_model)
    base_model = _base_model(input_shape, weights=weights, trainable=weights is None or base_trainable)
    
    embedding = base_model(image, training=(None if weights is None else False))
    normalized_embedding = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embedding)

    # Create unique projections for each variant type
    assert npsv2_pb2.StructuralVariant.Type.values() == list(range(len(npsv2_pb2.StructuralVariant.Type.values()))), "SV type encodings aren't contiguous range starting at 0, as required by tf.case"
    projections = []
    for _ in npsv2_pb2.StructuralVariant.Type.keys() if typed_projection else range(1):
        projection = normalized_embedding if normalize_embedding else embedding
        if len(projection_size) > 0:
            if stop_gradient_before_projection:
                projection = tf.stop_gradient(projection)
            
            # Enable MLP (like supcon or SimCLR) where intermediate layers use ReLU
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
        projections.append(projection)

    if typed_projection:
        # Eliminate parameters for types we aren't considering, by setting those values to 0
        if selected_type_indices:
            for i in (set(npsv2_pb2.StructuralVariant.Type.values()) - set(selected_type_indices)):
                projections[i] = layers.Lambda(lambda x: tf.zeros(tf.shape(x)))(projections[selected_type_indices[0]])

        # To use TimeDistributed, use a single input and output (all of the projection heads)
        projection = layers.Lambda(lambda x: tf.stack(x, axis=1))(projections)
    else:
        projection = projections[0]

    return tf.keras.Model(inputs=image, outputs=[embedding, normalized_embedding, projection], name="encoder")


def _sparse_binarize_genotypes(y_true, y_pred):
    # Adapted from: https://github.com/keras-team/keras/blob/372515ca4540d99eece842ab0b56d13eedf37b7c/keras/metrics.py#L3571
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true)
    y_pred_rank = y_pred.shape.ndims
    y_true_rank = y_true.shape.ndims
    # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
    if (y_true_rank is not None) and (y_pred_rank is not None) and (len(tf.keras.backend.int_shape(y_true)) == len(tf.keras.backend.int_shape(y_pred))):
        y_true = tf.squeeze(y_true, [-1])
    
    # Binarize genotypes to reference/non-reference
    y_pred = tf.argmax(y_pred, axis=-1) != 0
    y_true = y_true != 0

    # If the predicted output and actual output types don't match, force cast them to match.
    if y_pred.dtype != y_true.dtype:
        y_pred = tf.dtypes.cast(y_pred, y_true.dtype)

    return y_true, y_pred

def sparse_nonref_genotype_concordance(y_true, y_pred):
    y_true, y_pred = _sparse_binarize_genotypes(y_true, y_pred)
    return tf.cast(tf.equal(y_true, y_pred), tf.keras.backend.floatx())


class GenotypingModel:    
    def summary(self):
        self._model.summary()

    def save(self, model_path: str):
        # We only save the encoder weights
        encoder = self._model.get_layer("encoder")
        encoder.save_weights(model_path)

    def _fit(self, cfg, training_dataset, validation_dataset=None):
        callbacks = []

        initial_epoch = 0
        
        # Load most recent checkpoint, extracting epoch number to restart training at specific epoch
        if cfg.training.checkpoint_dir:
            logging.info("Saving checkpoints to %s", cfg.training.checkpoint_dir)

            restart_path = os.path.join(cfg.training.checkpoint_dir, "restart")    
            backup_callback = tf.keras.callbacks.BackupAndRestore(restart_path)
            callbacks.append(backup_callback)

            if cfg.training.save_best_validation_checkpoint and validation_dataset:
                # TODO: Integrate the two checkpoint callbacks, so it keeps track of latest and best
                best_path = os.path.join(cfg.training.checkpoint_dir, "val_accuracy", "best")
                best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=best_path,
                    save_weights_only=True,
                    save_best_only=True,
                    monitor="val_genotypes_sparse_categorical_accuracy",
                    mode="max",
                    verbose=1,
                )
                callbacks.append(best_checkpoint_callback)


        if cfg.training.log_dir:
            log_dir = os.path.join(cfg.training.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            logging.info("Logging TensorBoard data to: %s", log_dir)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False)
            callbacks.append(tensorboard_callback)

        if cfg.training.steps_per_epoch is not None:
            training_dataset = training_dataset.repeat()

        self._model.fit(
            training_dataset,
            validation_data=validation_dataset,
            initial_epoch=initial_epoch,
            epochs=cfg.training.epochs,
            steps_per_epoch=cfg.training.steps_per_epoch,
            callbacks=callbacks,
            validation_freq=cfg.training.validation_freq,
        )

        if cfg.training.checkpoint_dir and cfg.training.save_best_validation_checkpoint and validation_dataset:
            logging.info("Loading best model from: %s", best_path)
            self._model.load_weights(best_path)



class SupervisedBaselineModel(GenotypingModel):
    def __init__(self, image_shape, replicates, model_path: str=None, **kwargs):
        self.image_shape = image_shape
        self._model = self._create_model(image_shape, **kwargs)
        if model_path:
            # Load all of the model weights, not just the encoder so we get the top layer
            self._model.load_weights(model_path)

    def save(self, model_path: str):
        self._model.save_weights(model_path)

    def _create_model(self, image_shape, model_path: str=None,  weights=None, base_trainable=True, **kwargs):
        encoder = _contrastive_encoder(
            image_shape,
            weights=weights,
            base_trainable=base_trainable,
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
    
            return tf.data.Dataset.from_tensors(({ 
                "query": tf.image.convert_image_dtype(features["image"], dtype=tf.float32),
            }, {
                "genotypes_logits": original_label,
                "genotypes": original_label,
            }))
        
        def _filter_missing_labels(features, original_label):
            return original_label is not None

        # Interleave SVs into the batch
        return (
            dataset
            .filter(_filter_missing_labels)
            .shuffle(cfg.training.shuffle, reshuffle_each_iteration=True)
            .interleave(_variant_to_training, cycle_length=cfg.training.variants_per_batch, num_parallel_calls=cfg.threads)
            .batch(cfg.training.variants_per_batch)
            .prefetch(tf.data.experimental.AUTOTUNE)  # Newer versions: tf.data.AUTOTUNE
        )

    def fit(self, cfg, training_dataset, validation_dataset=None):
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.training.learning_rate),
            loss={ "genotypes_logits": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) },
            metrics={ "genotypes": ["sparse_categorical_accuracy", sparse_nonref_genotype_concordance] },
        )
                
        self._fit(
            cfg,
            self._train_input(cfg, training_dataset),
            validation_dataset=validation_dataset and self._test_input(cfg, validation_dataset)
        )

    def _test_input(self, cfg, dataset, max_replicates=sys.maxsize):
        def _variant_to_test(features, original_label):
            return ({
                "query": tf.image.convert_image_dtype(features["image"], dtype=tf.float32),
            }, {
                "genotypes_logits": original_label,
                "genotypes": original_label,
            })

        return (
            dataset
            .map(_variant_to_test)
            .batch(cfg.training.variants_per_batch)
            .prefetch(tf.data.experimental.AUTOTUNE)  # Newer versions: tf.data.AUTOTUNE
        )

    def predict(self, cfg, dataset):
        return self._model.predict(self._test_input(cfg, dataset))


class SimulatedEmbeddingsModel(GenotypingModel):
    def __init__(self, image_shape, replicates, model_path: typing.Union[str, typing.List[str]]=None, **kwargs):
        self.image_shape = image_shape
        self._model = self._create_model(image_shape, model_path=model_path, **kwargs)

    def _create_model(self, image_shape, model_path: typing.Union[str, typing.List[str]]=None, projection_size=[512], weights=None, typed_projection=False, base_trainable=True, **kwargs):
        assert len(image_shape) == 3, "Model only supports single images"
        
        support = layers.Input((3,) + image_shape, name="support")
        query = layers.Input(image_shape, name="query")
        svtype = layers.Input((), name="svtype", dtype=tf.int32)

        def siamese_network(model_path, index=""):
             # In this context, we only care about the projection output
            encoder =_contrastive_encoder(image_shape, weights=weights, base_trainable=base_trainable, normalize_embedding=False, projection_size=projection_size, batch_normalize_projection=True, normalize_projection=True, typed_projection=typed_projection, **kwargs)
            _, _, projection = encoder.output
            encoder = tf.keras.Model(encoder.input, projection, name=f"encoder{index}")
            if model_path:
                encoder.load_weights(model_path)

            support_embeddings = layers.TimeDistributed(encoder, name=f"support_embeddings{index}")(support)
            query_embeddings = encoder(query)
            
            if typed_projection:
                # Because of limitations in TimeDistributed we can't move this into the encoder itself and instead
                # implement it here so there is only one input and output to TimeDistributed
                def _chooser(inputs, axis=None):
                    # Slice out which projection to use for this particular variant type. Assumes select, e.g. type enumeration indexes,
                    # are valid indices for gather
                    select, embeddings = inputs
                    return tf.gather(embeddings, select, batch_dims=1, axis=axis)

                support_embeddings = layers.Lambda(lambda x: _chooser(x, axis=2))([svtype, support_embeddings])
                query_embeddings = layers.Lambda(lambda x: _chooser(x))([svtype, query_embeddings])            
            
            distances = layers.Lambda(_query_distances, name=f"distances{index}")([query_embeddings, support_embeddings])
            return distances

        if isinstance(model_path, collections.abc.Sized) and len(model_path) > 1:
            # Average the distance outputs of the ensemble members together
            distance_layers = []
            for i, path in enumerate(model_path):
                distance_layers.append(siamese_network(path, index=i))
            distances = layers.Average(name="distances")(distance_layers)
        else:
            distances = siamese_network(as_scalar(model_path))

        # Convert distance to probability
        genotypes = layers.Lambda(lambda x: tf.nn.softmax(-x, axis=-1), name="genotypes")(distances) 

        inputs = [query, support]
        if typed_projection:
            inputs.append(svtype)
        return tf.keras.Model(inputs=inputs, outputs=[genotypes, distances])


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
            return contrastive_loss(y_true, y_pred, margin=cfg.training.contrastive_margin) * weights

        # TODO: Add precision/recall
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.training.learning_rate),
            loss={ "distances": _loss_wrapper },
            metrics={ "genotypes": ["sparse_categorical_accuracy", sparse_nonref_genotype_concordance] },
        )
        
        # Use a maximum of 1 replicate for validation
        self._fit(
            cfg,
            self._train_input(cfg, training_dataset),
            validation_dataset=validation_dataset and self._test_input(cfg, validation_dataset, max_replicates=1),
        )


    def _test_input(self, cfg, dataset, batch_size):
        def _variant_to_test(features, original_label):
            #one_hot_label = original_label if original_label is not None else -1
            return ({
                "query": tf.image.convert_image_dtype(features["image"], dtype=tf.float32),
                "support": tf.image.convert_image_dtype(features["sim/images"][:,0], dtype=tf.float32),
            }, {
                "distances": tf.one_hot(original_label, depth=3, dtype=tf.float32),
                "genotypes": original_label,
            })

        return dataset.map(_variant_to_test).batch(batch_size)


    def predict(self, cfg, dataset):
        return self._model.predict(self._test_input(cfg, dataset))


class JointEmbeddingsModel(SimulatedEmbeddingsModel):
    def __init__(self, image_shape, replicates, model_path: typing.Union[str, typing.List[str]]=None, **kwargs):
        super().__init__(image_shape, replicates, model_path=model_path, **kwargs)

        file_descriptor_set = descriptor_pb2.FileDescriptorSet()
        npsv2_pb2.DESCRIPTOR.CopyToProto(file_descriptor_set.file.add())
        self._descriptor_source = b'bytes://' + file_descriptor_set.SerializeToString()
    
    def _variant_to_dataset(self, cfg, features, original_label, max_replicates=sys.maxsize):
        support_shape = tf.shape(features["sim/images"])
        support_replicates = tf.math.minimum(support_shape[1], min(cfg.simulation.replicates, max_replicates))
        tf.debugging.assert_greater(support_replicates, 0)
        
        # TODO: Actually obtain reference size (since that determines the image size)
        _, [_, svtype] = tf.io.decode_proto(
            features["variant/encoded"],
            "npsv2.StructuralVariant",
            ["svlen", "svtype"],
            [tf.int64, tf.int32], # svtype is an enum
            descriptor_source=self._descriptor_source
        )
        svtype = tf.squeeze(svtype)

        queries = tf.image.convert_image_dtype(features["image"], dtype=tf.float32)
        queries = tf.repeat(tf.expand_dims(queries, axis=0), repeats=support_replicates, axis=0)

        # Swap AC and replicates dimensions, so we can pass triples of all genotypes to the model. Slice to control
        # the number of replicates used per variant in training
        support = tf.image.convert_image_dtype(features["sim/images"][:,:support_replicates], dtype=tf.float32)
        support = tf.transpose(support, perm=(1, 0) + tuple(range(2, 2+len(self.image_shape))))

        inputs = { 
            "query": queries,
            "support": support,
        }
        if cfg.model.typed_projection:
            inputs["svtype"] = tf.repeat(svtype, support_replicates)

        if original_label is not None:
            return tf.data.Dataset.from_tensor_slices((inputs, {
                "distances": tf.tile(tf.one_hot([original_label], depth=3, dtype=tf.float32), (support_replicates, 1)),
                "genotypes": tf.repeat(original_label, support_replicates),
            }))
        else:
            return tf.data.Dataset.from_tensor_slices((inputs, None))

    def _train_input(self, cfg, dataset):
        def _variant_to_training(features, original_label):
            return self._variant_to_dataset(cfg, features, original_label)
        
        def _filter_missing_labels(features, original_label):
            return original_label is not None

        # Interleave SVs from different variants into the batch
        return (
            dataset
            .filter(_filter_missing_labels)
            .shuffle(cfg.training.shuffle, reshuffle_each_iteration=True)
            .interleave(_variant_to_training, cycle_length=cfg.training.variants_per_batch, num_parallel_calls=cfg.threads)
            .batch(cfg.training.variants_per_batch)
            .prefetch(tf.data.experimental.AUTOTUNE)  # Newer versions: tf.data.AUTOTUNE
        )

    def _test_input(self, cfg, dataset, max_replicates=sys.maxsize):
        def _variant_to_test(features, original_label):
            return self._variant_to_dataset(cfg, features, original_label, max_replicates=max_replicates)

        return (
            dataset
            .flat_map(_variant_to_test)
            .batch(cfg.training.variants_per_batch)
            .prefetch(tf.data.experimental.AUTOTUNE)  # Newer versions: tf.data.AUTOTUNE
        )

