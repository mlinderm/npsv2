"""Siamese Model"""
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import Model


class DistanceLayer(layers.Layer):
    """This layer is responsible for computing the distance between the anchor embedding
    and the simulation embedding. This layer does not have any trainable weights. 
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, simulation):
        """Compute the euclidean distance between the anchor and simulation vector"""

        distance = tf.norm(anchor - simulation, axis=-1, ord="euclidean")
        return distance


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the contrastive loss usng the two embeddings produced by the Siamese Network.

    The constrastive loss is defined as:
      Y * D^2 + (1 - Y) * max(margin - D, 0)^2

    Model Inputs:
      Real(anchor) image: (100, 300, 6).
      Simulated hom.ref image: (100, 300, 6).
    
    Model Outputs:
      Distance: float distance between the real and simulated embeddings.
      anchor_embd: (1, 512) anchor image embeddings.
      sim_embd: (1, 512) simulated image embeddings.
    """
    def __init__(self, target_shape, margin=0.5):
        super().__init__()
        self.target_shape = target_shape
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")
        self.siamese_network = self.build_siamese()
        # the _set_inputs methods make sure model.save_weights can function correctly
        self._set_inputs(inputs=self.siamese_network.inputs, outputs=self(self.siamese_network.inputs))

    def call(self, inputs):
        """Return the siamese network defined on the given input.

        Args:
            inputs: the raw inputs from the dataset.

        Returns:
            constructed siamese network.
        """
        return self.siamese_network(inputs)

    def train_step(self, data):
        """Custom defined training step logic"""
        x, y = data # (images, labels)
        # use GradientTape to record and compute losses
        with tf.GradientTape() as tape:
            distance, _, _ = self.siamese_network(x)
            loss = self._compute_loss(y["label"], distance)

        # storing the gradient of the loss function
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
        }

    def test_step(self, data):
        """Custom defined validation step logic"""
        x, y = data # (images, labels)
        distance, _, _ = self.siamese_network(x)
        loss = self._compute_loss(y["label"], distance)

        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
        }

    def _compute_loss(self, y_true, y_pred):
        """Return the contrastive loss calculated using the true and predicted label.

        Args:
          y_true: binary labels.
          y_pred: distances between two embedding matrices.
        
        Returns:
          Computed loss.
        """
        return tfa.losses.contrastive_loss(y_true, y_pred)

    def convolutional_block(self, inputs):
        """Defines the convolutional layers of the siamese network"""
        x = layers.Conv2D(32, 3, padding = 'valid', activation = 'relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(2)(x)
        
        x = layers.Conv2D(32, 3, padding = 'valid', activation = 'relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(2)(x)
        
        x = layers.Conv2D(64, 3, padding = 'same', activation = 'relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(2)(x)
            
        x = layers.Conv2D(64, 3, padding = 'same', activation = 'relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(2)(x)

        x = layers.Conv2D(128, 5, padding = 'same', activation = 'relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(2)(x)

        x = layers.Conv2D(128, 5, padding = 'same', activation = 'relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(2)(x)
        
        return x

    def build_siamese(self):
        """Defines the model architecture of the siamese network. The model consist of a 
        covolutional block that takes the the real and simulation image and map them as 
        their corresponding feature map. Some fully connected layers then maps the two
        feature maps into a joint embedding space. Finally, the distance layer calculates
        the distance between the two vectors.
        """
        # build embedding layer
        inputs = tf.keras.Input(self.target_shape, name="image")
        base_cnn = self.convolutional_block(inputs)

        flatten = layers.Flatten()(base_cnn)
        dense1 = layers.Dense(1024, activation="relu")(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(512, activation="relu")(dense1)
        dense2 = layers.Dropout(0.2)(dense2)
        dense2 = layers.BatchNormalization()(dense2)
        embedding = layers.Dense(512)(dense2)
        output = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embedding)
        embedding = Model(inputs, output, name="Embedding")

        # build distance layer
        anchor_input = layers.Input(name="anchor", shape=self.target_shape)
        simulation_input = layers.Input(name="sim", shape=self.target_shape)

        anchor_embd = embedding(anchor_input)
        sim_embd = embedding(simulation_input)

        distance_layer = DistanceLayer()
        distance = distance_layer(anchor_embd, sim_embd)

        siamese_network = Model(
            inputs=[anchor_input, simulation_input], outputs=[distance, anchor_embd, sim_embd]
        )
        return siamese_network

    @property
    def metrics(self):
        return [self.loss_tracker]


