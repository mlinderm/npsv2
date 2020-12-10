import argparse, itertools, random
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from npsv2.images import load_example_dataset

# Adapted from: https://github.com/few-shot-learning/Keras-FewShotLearning/blob/master/keras_fsl/models/encoders/koch_net.py
def siamese_networks_model(input_shape):
    encoder = tf.keras.models.Sequential(
        [
            layers.Input(input_shape),
            layers.Conv2D(
                64,
                (10, 10),
                activation=tf.nn.relu,
                kernel_regularizer=l2(),
                kernel_initializer=RandomNormal(0.0, 0.01),
                bias_initializer=RandomNormal(0.5, 0.01),
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(
                128,
                (7, 7),
                activation=tf.nn.relu,
                kernel_regularizer=l2(),
                kernel_initializer=RandomNormal(0.0, 0.01),
                bias_initializer=RandomNormal(0.5, 0.01),
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(
                128,
                (4, 4),
                activation=tf.nn.relu,
                kernel_regularizer=l2(),
                kernel_initializer=RandomNormal(0.0, 0.01),
                bias_initializer=RandomNormal(0.5, 0.01),
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(
                256,
                (4, 4),
                activation=tf.nn.relu,
                kernel_regularizer=l2(),
                kernel_initializer=RandomNormal(0.0, 0.01),
                bias_initializer=RandomNormal(0.5, 0.01),
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(
                4096,
                activation=tf.nn.sigmoid,
                kernel_initializer=RandomNormal(0.0, 0.2),
                bias_initializer=RandomNormal(0.5, 0.01),
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



params = argparse.Namespace(learning_rate=0.004, l2=0.001, batch_size=32, total_epochs=20)

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# Decay learning rate, e.g. 
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=params.learning_rate,
#     decay_steps=10000, # e.g. set to the number of steps per epoch
#     decay_rate=0.99)

optimizer = tf.keras.optimizers.Adam(lr=params.learning_rate)
model = siamese_networks_model((300, 300, 1))
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.binary_crossentropy,
    metrics=["binary_accuracy"],
)
model.summary()


def random_pair_indices(replicates, pairs_per_variant):
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


def variant_to_training_pairs(features, original_label):
    replicates = features["sim/0/images"].shape[0]

    images = tf.stack([features["sim/0/images"], features["sim/1/images"], features["sim/2/images"]])
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)

    (query_indices, support_indices), pair_labels = random_pair_indices(replicates, 8)

    image_tensors = {
        "query": tf.gather_nd(images, query_indices),
        "support": tf.gather_nd(images, support_indices),
    }
    label_tensor = tf.constant(pair_labels, dtype=tf.uint64)

    # Construct data to permit "flat_map" and thus more flexible batching downstream
    return tf.data.Dataset.from_tensor_slices((image_tensors, label_tensor))


example_dataset = load_example_dataset(
    "/storage/mlinderman/projects/sv/testing/npsv2/images/300x300.tfrecords.gz", 
    #"/storage/mlinderman/projects/sv/testing/npsv2/test.tfrecords",
    with_label=True, 
    with_simulations=True,
)
train_dataset = example_dataset.flat_map(variant_to_training_pairs).batch(16)

history = model.fit(train_dataset, epochs=5)

# TODO: Add early stopping callback

model.save('models/siamese_network') 


def variant_to_test_pairs(features, original_label):
    query_images = tf.stack([tf.image.convert_image_dtype(features["image"], dtype=tf.float32)] * 3)
    support_images = tf.image.convert_image_dtype(
        tf.stack([features[f"sim/{ac}/images"][0, :, :, :] for ac in range(3)]), dtype=tf.float32
    )

    image_tensors = {
        "query": query_images,
        "support": support_images,
    }

    return image_tensors, original_label


test_dataset = example_dataset.map(variant_to_test_pairs)

y_true = tf.concat(list(test_dataset.map(lambda _, label: label)), 0)
y_pred = tf.reshape(model.predict(test_dataset), (-1, 3))

results = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
print(f"Final test accuracy: {np.mean(results)}")



# for example in test:
#     print("test") #,example)
# print(example_dataset.take(1))
# train_ds = (
#     load_example_dataset(
#         params,
#         "/storage/mlinderman/projects/sv/testing/npsv2/images/300x300.tfrecords",
#         with_labels=True,
#     )
#     .shuffle(buffer_size=8192, reshuffle_each_iteration=True)
#     .batch(batch_size=params.batch_size)
#     .repeat(1)
# )


# model.fit(example_dataset, epochs=5)

# l2_reg = tf.keras.regularizers.l2

# # model = tf.keras.models.Sequential(
# #     [
# #         tf.keras.layers.Conv2D(
# #             filters=32,
# #             kernel_size=(3, 3),
# #             activation=tf.nn.relu,
# #             kernel_regularizer=l2_reg(params.l2),
# #         ),
# #         tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
# #         tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
# #         tf.keras.layers.Dropout(rate=0.25),
# #         tf.keras.layers.Flatten(),
# #         tf.keras.layers.Dense(units=3, activation="softmax"),
# #     ]
# # )

# # Adapt model from Nucleus demo (to 2-D)
# model = tf.keras.models.Sequential(
#     [
#         tf.keras.layers.Conv2D(
#             filters=16,
#             kernel_size=(5, 5),
#             activation=tf.nn.relu,
#             kernel_regularizer=l2_reg(params.l2),
#             input_shape=(300,300,1)
#         ),
#         tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
#         tf.keras.layers.Conv2D(
#             filters=16,
#             kernel_size=(3, 3),
#             activation=tf.nn.relu,
#             kernel_regularizer=l2_reg(params.l2),
#         ),
#         tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(
#           units=16,
#           activation=tf.nn.relu,
#           kernel_regularizer=l2_reg(params.l2)),
#         tf.keras.layers.Dropout(rate=0.3),
#         tf.keras.layers.Dense(
#           units=16,
#           activation=tf.nn.relu,
#           kernel_regularizer=l2_reg(params.l2)),
#         tf.keras.layers.Dropout(rate=0.3),
#         tf.keras.layers.Dense(units=3, activation="softmax"),
#     ]
# )


# optimizer = tf.keras.optimizers.Adam(lr=params.learning_rate)
# model.compile(
#     optimizer=optimizer,
#     loss=tf.keras.losses.sparse_categorical_crossentropy,
#     metrics=["accuracy"],
# )


# train_ds = (
#     load_example_dataset(
#         params,
#         "/storage/mlinderman/projects/sv/testing/npsv2/images/300x300.tfrecords",
#         with_labels=True,
#     )
#     .shuffle(buffer_size=8192, reshuffle_each_iteration=True)
#     .batch(batch_size=params.batch_size)
#     .repeat(1)
# )


# model.fit(train_ds, epochs=params.total_epochs)
# print(model.summary())

# test_ds = load_example_dataset(
#     params,
#     "/storage/mlinderman/projects/sv/testing/npsv2/images/300x300.tfrecords.test",
#     with_labels=True,
# )

# test_metrics = model.evaluate(test_ds.batch(batch_size=params.batch_size), verbose=0)
# print(f"Final test metrics - loss: {test_metrics[0]}- accuracy: {test_metrics[1]}")
