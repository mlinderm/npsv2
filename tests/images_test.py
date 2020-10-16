import argparse, io, os, tempfile, unittest
from unittest.mock import patch

from nucleus.io import vcf
import tensorflow as tf
from PIL import Image
from npsv2 import images

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")


class CreateSingleImageTest(unittest.TestCase):
    def setUp(self):
        vcf_path = os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")
        with vcf.VcfReader(vcf_path) as reader:
            self.variant = next(reader)

        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")
        self.params = argparse.Namespace()

    def test_resizing_image(self):
        example = images.create_single_example(self.params, self.variant, self.bam_path, "1:899722-900192")
        self.assertNotEqual(images._example_shape(example), (300, 300, 1))

        example = images.create_single_example(self.params, self.variant, self.bam_path, "1:899722-900192", image_shape=(300,300))
        self.assertEqual(images._example_shape(example), (300, 300, 1))

    def test_label_extraction(self):
        example = images.create_single_example(self.params, self.variant, self.bam_path, "1:899822-900092")
        self.assertFalse("label" in example.features.feature)
        
        example = images.create_single_example(self.params, self.variant, self.bam_path, "1:899722-900192", label=1)
        self.assertEqual(images._example_label(example), 1)

    def test_example_to_image(self):
        with tempfile.TemporaryDirectory() as tempdir:
            png_path = os.path.join(tempdir, "test.png")

            example = images.create_single_example(self.params, self.variant, self.bam_path, "1:899822-900092", image_shape=(300,300))
            images.example_to_image(example, png_path)
            
            self.assertTrue(os.path.exists(png_path))
            with Image.open(png_path) as image:
                self.assertEqual(image.size, (300, 300))


class VCFExampleGenerateTest(unittest.TestCase):
    def setUp(self):
        self.params = argparse.Namespace()
        self.vcf_path = os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")
        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")

    def test_vcf_generator_runs_without_error(self):
        all_examples = images.make_vcf_examples(self.params, self.vcf_path, self.bam_path)
        self.assertEqual(len(list(all_examples)), 1)

    def test_dataset_roundtrip(self):
        with tempfile.TemporaryDirectory() as tempdir:
            dataset_path = os.path.join(tempdir, "test.tfrecord")
            with tf.io.TFRecordWriter(dataset_path) as dataset:
                all_examples = images.make_vcf_examples(self.params, self.vcf_path, self.bam_path, image_shape=(300, 300), sample_or_label="HG002")
                for example in all_examples:
                    self.assertEqual(images._example_label(example), 2)
                    dataset.write(example.SerializeToString())
            self.assertTrue(os.path.exists(dataset_path))
            
            dataset = images.load_example_dataset(self.params, dataset_path, with_labels=True)
            for image, label in dataset:
                self.assertEqual(image.shape, (300, 300, 1))
                self.assertEqual(label, 2)

