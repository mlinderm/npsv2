# MIT License
#
# Copyright (c) 2021 Michael Schuster
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging, shutil, os
from collections import deque

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import tf_utils

# Adapted from https://github.com/schustmi/tf_utils/blob/main/tf_utils/callbacks.py
class NModelCheckpoint(ModelCheckpoint):
    """Callback to only save the most recent N checkpoints of a keras model or its weights.

    Arguments:
        filepath: string or `PathLike`, path where the checkpoints will be saved. Make sure
          to pass a format string as otherwise the checkpoint will be overridden each step.
          (see `keras.callbacks.ModelCheckpoint` for detailed formatting options)
        max_to_keep: int, how many checkpoints to keep.
        existing_checkpoints: iterable, existing checkpoints to track
        **kwargs: Additional arguments that get passed to `keras.callbacks.ModelCheckpoint`.
    """
    def __init__(self, filepath, max_to_keep: int, existing_checkpoints=[], **kwargs):
        if kwargs.pop("save_best_only", None):
            logging.warning("Setting `save_best_only` to False.")

        if max_to_keep < 1:
            logging.warning("Parameter `max_to_keep` must be greater than 0, setting it to 1.")
            max_to_keep = 1

        super().__init__(filepath, save_best_only=False, **kwargs)
        self._keep_count = max_to_keep
        self._checkpoints = deque(existing_checkpoints)

    def _save_model(self, epoch, logs):
        super()._save_model(epoch, logs)
        logs = tf_utils.to_numpy_or_python_type(logs or {})
        filepath = self._get_file_path(epoch, logs)

        if not self._checkpoint_exists(filepath):
            # Did not save a checkpoint for current epoch
            return

        self._checkpoints.append(filepath)
        while len(self._checkpoints) > self._keep_count:
            self._delete_checkpoint_files(self._checkpoints.popleft())

    @staticmethod
    def _delete_checkpoint_files(checkpoint_path):
        logging.info(f"Removing files for checkpoint '{checkpoint_path}'")

        if os.path.isdir(checkpoint_path):
            # For SavedModel format delete the whole directory
            shutil.rmtree(checkpoint_path)
            return

        for f in tf.io.gfile.glob(checkpoint_path + "*"):
            os.remove(f)
