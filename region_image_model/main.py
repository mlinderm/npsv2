from models.siamese_model import SiameseModel
from eval_model import *
from TR_Region_Image import load_dataset, load_single_dataset

import hydra
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from tabulate import tabulate


def variant_to_training(features, labels):
    """Generate training label for each anchor-simulation image pair. If original is hom. ref label same class(1), 
    otherwise label mismatch(0)
    """
    # if label == 0 -> hom.ref -> match with hom ref simulation -> training label = 1
    label = tf.cond(tf.math.equal(labels["label"], tf.constant(0, dtype=tf.int64)), lambda: 1, lambda: 0)
    labels = {
        "label": label
    }
    return features, labels


def optimize_for_training(cfg, ds):
    """Filter out non-training relevant labels and batch the dataset for training

    Args:
      config: Bunch dict. Dictionary with all training parameters. 
      ds: tf.data.Dataset. Dataset containing images and variant related information.
    """
    return (
        ds
        .map(variant_to_training, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(cfg.mode.batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )


def train_siamese(cfg):
    """Train a siamese network on data in tfrecord file(s) according to the parameters in config and save 
    the data in result path specified in cfg.

    Args:
      config: dictionary contain all the training parameters.
    """
    target_shape = (100, 300, 6)
    train_dataset, val_dataset_all = load_dataset(
        cfg.mode.training_storage_dir,
        cfg.mode.training_storage_subdir,
        cfg.mode.sample_names,
        cfg.mode.shuffle_buffer,
        subset=cfg.mode.subset
    )

    train_dataset = optimize_for_training(cfg, train_dataset)
    val_dataset = optimize_for_training(cfg, val_dataset_all)

    model = SiameseModel(target_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.mode.learning_rate))
    model.fit(train_dataset, epochs=cfg.mode.epochs, validation_data=val_dataset)
    
    assert model.save_spec() is not None
    try:
        model.save_weights(cfg.mode.model_path)
    except ValueError as e:
        print(f'{type(e).__name__}: ', *e.args)

    print(f"model saved at {cfg.mode.model_path}")


def predict_on_eval(cfg):
    """Test a trained model according to the parameters and output path specified in config file"""
    target_shape = (100, 300, 6)
    test_tfrecords_path = os.path.join(cfg.mode.eval_storage_dir, "images.tfrecords.gz")
    val_dataset_all = load_single_dataset(test_tfrecords_path, have_id=True)

    val_dataset = optimize_for_training(cfg, val_dataset_all)

    #model = tf.keras.models.load_model(cfg.mode.model_path)
    model = SiameseModel(target_shape)
    model.load_weights(cfg.mode.model_path)
    save_pred(cfg, model, val_dataset, val_dataset_all, have_id=True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if cfg.mode.command == "train":
        train_siamese(cfg)
    elif cfg.mode.command == "eval":
        # run test on test sets, defaults to HG002
        df = predict_on_eval(cfg)
    elif cfg.mode.command == "refine":
        refine_results(cfg)


if __name__ == "__main__":
    main()