"""TR Region Image Dataset."""

import tensorflow as tf
from npsv2 import images
import numpy as np
from npsv2 import npsv2_pb2
from google.protobuf import descriptor_pb2
from tqdm import tqdm
import os


def samples_to_filenames(storage_dir, storage_subdir, sample_names: list) -> list:
    """Convert sample names to training files that should be its corresponding image tfrecord"""
    filenames = []
    for sample in sample_names:
        tfrecords_path = os.path.join(storage_dir, sample, storage_subdir, "images.tfrecords.gz")
        if not os.path.exists(tfrecords_path): 
            print(f"missing tfrecords file in {sample}")
            continue
        filenames.append(tfrecords_path)

    return filenames


def load_single_dataset(filename: str, have_id: bool = False) -> tf.data.Dataset:
    """Load single dataset specified by the filename.

    Args:
        filename: String. the name of the file that contains the training data
          generated through the region image generator and should have
          the format .tfrecords.gz.
    """

    def parse_tfrecord_fn(example):
        """Describe the partition scheme for a single entry in the tfrecord file.

        example: should be an entry from tf.TFRecordDataset
        """
        # Description to parse SV protobuf saved with embeddings
        file_descriptor_set = descriptor_pb2.FileDescriptorSet()
        npsv2_pb2.DESCRIPTOR.CopyToProto(file_descriptor_set.file.add())
        descriptor_source = b"bytes://" + file_descriptor_set.SerializeToString()

        feature_description = {
            "image/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            "sim/images/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            "variant/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            "label": tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
        }

        if have_id:
            feature_description["id/encoded"] = tf.io.FixedLenFeature(shape=(), dtype=tf.string)

        example = tf.io.parse_single_example(example, feature_description)
        example["image"] = tf.io.parse_tensor(
            example["image/encoded"], tf.uint8
        )
        example["sim"] = tf.io.parse_tensor(
            example["sim/images/encoded"], tf.uint8
        )
        _, [contig, start, svlen] = tf.io.decode_proto(
            example["variant/encoded"],
            "npsv2.StructuralVariant",
            ["contig", "start", "svlen"],
            [tf.string, tf.int64, tf.int64],
            descriptor_source=descriptor_source,
        )
        example["contig"] = contig
        example["start"] = start
        example["svlen"] = svlen
        if have_id:
            example["id"] = tf.io.parse_tensor(example["id/encoded"], tf.string)
        return example

    def prepare_sample(feature_dic):
        """Parse tfrecord file into training dataset.

        Args:
        feature_fic: should be the output of parse_tfrecord_fn
        """
        features = {
            "anchor": feature_dic["image"],
            "sim": tf.squeeze(feature_dic["sim"]) # reduce the original size (1, 1, 100, 300, 6) to (100, 300, 6)
        }
        labels = {
            "label": tf.reshape(feature_dic["label"], shape=(1,)), # enforce shape
            "contig": feature_dic["contig"],
            "start": feature_dic["start"],
            "svlen": feature_dic["svlen"],
        }
        if have_id:
            labels["id"] = feature_dic["id"][0]
        return features, labels

    dataset = (
        tf.data.TFRecordDataset(filename, compression_type="GZIP")
        .map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .map(prepare_sample, num_parallel_calls=tf.data.AUTOTUNE)
    )
    return dataset


def load_dataset(storage_dir, storage_subdir, sample_names, shuffle_buffer, subset=0) -> tf.data.Dataset:
    """Load the TR Region Image dataset generated from HGSVC2 sv call set.

    This is a dataset of (sample num)  100 * 300 * 6 images each representing a TR
    region surrounding a Structural Variant of 2 genotype class along with related metadata. 

    The classes are:
      Label 0: Hom. Ref
      Label 1: Hom. Alt and Heterozygous
    
    The metadata are:
      contig: The contig of the SV
      start: The starting position of the SV
      len: The length of the SV

    Args:
      storage_dir:
      storage_subdir:
      sample_names:
      shuffle_buffer:
      subset: (optional) 
    """
    filenames = samples_to_filenames(storage_dir, storage_subdir, sample_names)
    dataset = (
        tf.data.Dataset.from_tensor_slices(filenames)
        .interleave(
            load_single_dataset,
            cycle_length=len(filenames),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    )
    if subset > 0:
        dataset = dataset.take(subset)
    return get_dataset_partitions(dataset, shuffle_buffer) 


#https://towardsdatascience.com/how-to-split-a-tensorflow-dataset-into-train-validation-and-test-sets-526c8dd29438
def get_dataset_partitions(ds, shuffle_buffer, train_split=0.9):
    assert train_split < 1
    
    if shuffle_buffer > 0:
        #Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=False)
    
    split_key = int(1 / (1 - train_split))

    def split_train_fn(dataset):
        dataset = (
            dataset.enumerate()
            .filter(lambda x,y: x % split_key != 0)
            .map(lambda x,y: y)
        )
        return dataset
    
    def split_val_fn(dataset):
        dataset = (
            dataset.enumerate()
            .filter(lambda x,y: x % split_key == 0)
            .map(lambda x,y: y)
        )
        return dataset
    
    return split_train_fn(ds), split_val_fn(ds)