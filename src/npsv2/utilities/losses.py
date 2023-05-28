# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Vendor contrastive loss to remove tensorflow-addons dependency. It is not available for aarch64,
is being deprecated."""

import tensorflow as tf

def is_tensor_or_variable(x):
    return tf.is_tensor(x) or isinstance(x, tf.Variable)

class LossFunctionWrapper(tf.keras.losses.Loss):
    """Wraps a loss function in the `Loss` class."""

    def __init__(
        self, fn, reduction=tf.keras.losses.Reduction.AUTO, name=None, **kwargs
    ):
        """Initializes `LossFunctionWrapper` class.

        Args:
          fn: The loss function to wrap, with signature `fn(y_true, y_pred,
            **kwargs)`.
          reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
              https://www.tensorflow.org/tutorials/distribute/custom_training)
            for more details.
          name: (Optional) name for the loss.
          **kwargs: The keyword arguments that are passed on to `fn`.
        """
        super().__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        """Invokes the `LossFunctionWrapper` instance.

        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.

        Returns:
          Loss values per sample.
        """
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = {}
        for k, v in iter(self._fn_kwargs.items()):
            config[k] = tf.keras.backend.eval(v) if is_tensor_or_variable(v) else v
        base_config = super().get_config()
        return {**base_config, **config}


@tf.function
def contrastive_loss(y_true, y_pred, margin = 1.0) -> tf.Tensor:
    r"""Computes the contrastive loss between `y_true` and `y_pred`.

    This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart at least
    by the margin constant for the samples of different labels.

    The euclidean distances `y_pred` between two embedding matrices
    `a` and `b` with shape `[batch_size, hidden_size]` can be computed
    as follows:

    >>> a = tf.constant([[1, 2],
    ...                 [3, 4],
    ...                 [5, 6]], dtype=tf.float16)
    >>> b = tf.constant([[5, 9],
    ...                 [3, 6],
    ...                 [1, 8]], dtype=tf.float16)
    >>> y_pred = tf.linalg.norm(a - b, axis=1)
    >>> y_pred
    <tf.Tensor: shape=(3,), dtype=float16, numpy=array([8.06 , 2.   , 4.473],
    dtype=float16)>

    <... Note: constants a & b have been used purely for
    example purposes and have no significant value ...>

    See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    Args:
      y_true: 1-D integer `Tensor` with shape `[batch_size]` of
        binary labels indicating positive vs negative pair.
      y_pred: 1-D float `Tensor` with shape `[batch_size]` of
        distances between two embedding matrices.
      margin: margin term in the loss definition.

    Returns:
      contrastive_loss: 1-D float `Tensor` with shape `[batch_size]`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    return y_true * tf.math.square(y_pred) + (1.0 - y_true) * tf.math.square(
        tf.math.maximum(margin - y_pred, 0.0)
    )


class ContrastiveLoss(LossFunctionWrapper):
    r"""Computes the contrastive loss between `y_true` and `y_pred`.

    This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart at least
    by the margin constant for the samples of different labels.

    See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    We expect labels `y_true` to be provided as 1-D integer `Tensor`
    with shape `[batch_size]` of binary integer labels. And `y_pred` must be
    1-D float `Tensor` with shape `[batch_size]` of distances between two
    embedding matrices.

    The euclidean distances `y_pred` between two embedding matrices
    `a` and `b` with shape `[batch_size, hidden_size]` can be computed
    as follows:

    >>> a = tf.constant([[1, 2],
    ...                 [3, 4],[5, 6]], dtype=tf.float16)
    >>> b = tf.constant([[5, 9],
    ...                 [3, 6],[1, 8]], dtype=tf.float16)
    >>> y_pred = tf.linalg.norm(a - b, axis=1)
    >>> y_pred
    <tf.Tensor: shape=(3,), dtype=float16, numpy=array([8.06 , 2.   , 4.473],
    dtype=float16)>

    <... Note: constants a & b have been used purely for
    example purposes and have no significant value ...>

    Args:
      margin: `Float`, margin term in the loss definition.
        Default value is 1.0.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply.
        Default value is `SUM_OVER_BATCH_SIZE`.
      name: (Optional) name for the loss.
    """

    def __init__(
        self,
        margin = 1.0,
        reduction: str = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = "contrastive_loss",
    ):
        super().__init__(
            contrastive_loss, reduction=reduction, name=name, margin=margin
        )