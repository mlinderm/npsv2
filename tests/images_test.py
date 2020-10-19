import argparse, io, os, tempfile, unittest
from unittest.mock import patch
import pysam
import tensorflow as tf
from PIL import Image
from npsv2.variant import Variant
from npsv2 import images

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")


class CreateSingleImageTest(unittest.TestCase):
    def setUp(self):
        record = next(pysam.VariantFile(os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")))
        self.variant = Variant.from_pysam(record)

        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")
        self.params = argparse.Namespace()

    def test_resizing_image(self):
        image_tensor = images.create_single_example(self.params, self.variant, self.bam_path, "1:899722-900192")
        self.assertNotEqual(image_tensor.shape, (300, 300, 1))

        image_tensor = images.create_single_example(
            self.params, self.variant, self.bam_path, "1:899722-900192", image_shape=(300, 300),
        )
        self.assertEqual(image_tensor.shape, (300, 300, 1))


class VCFExampleGenerateTest(unittest.TestCase):
    def setUp(self):
        self.params = argparse.Namespace()
        self.vcf_path = os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")
        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")

    def test_vcf_generator_runs_without_error(self):
        all_examples = images.make_vcf_examples(self.params, self.vcf_path, self.bam_path)
        self.assertEqual(len(list(all_examples)), 1)

    def test_label_extraction(self):
        example = next(images.make_vcf_examples(self.params, self.vcf_path, self.bam_path))
        self.assertFalse("label" in example.features.feature)

        example = next(images.make_vcf_examples(self.params, self.vcf_path, self.bam_path, sample_or_label="HG002"))
        self.assertEqual(images._example_label(example), 2)

        example = next(images.make_vcf_examples(self.params, self.vcf_path, self.bam_path, sample_or_label=1))
        self.assertEqual(images._example_label(example), 1)

    def test_example_to_image(self):
        all_examples = images.make_vcf_examples(self.params, self.vcf_path, self.bam_path, image_shape=(300, 300))
        with tempfile.TemporaryDirectory() as tempdir:
            png_path = os.path.join(tempdir, "test.png")

            images.example_to_image(next(all_examples), png_path)

            self.assertTrue(os.path.exists(png_path))
            with Image.open(png_path) as image:
                self.assertEqual(image.size, (300, 300))

    def test_dataset_roundtrip(self):
        with tempfile.TemporaryDirectory() as tempdir:
            dataset_path = os.path.join(tempdir, "test.tfrecord")
            # We would like to compress the records, but any compression leads to
            # tensorflow.python.framework.errors_impl.DataLossError: corrupted record at 0
            with tf.io.TFRecordWriter(dataset_path) as dataset:
                all_examples = images.make_vcf_examples(
                    self.params, self.vcf_path, self.bam_path, image_shape=(300, 300), sample_or_label="HG002",
                )
                for example in all_examples:
                    self.assertEqual(images._example_label(example), 2)
                    dataset.write(example.SerializeToString())
            self.assertTrue(os.path.exists(dataset_path))

            dataset = images.load_example_dataset(self.params, dataset_path, with_labels=True)
            for image, label in dataset:
                self.assertEqual(image.shape, (300, 300, 1))
                self.assertEqual(label, 2)

