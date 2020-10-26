import argparse, io, os, shutil, tempfile, unittest
from unittest.mock import patch, call
import pysam
import tensorflow as tf
import numpy as np
from PIL import Image
from npsv2.variant import Variant
from npsv2.range import Range
from npsv2 import images
from npsv2 import npsv2_pb2

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")


class CreateSingleImageTest(unittest.TestCase):
    def setUp(self):
        record = next(pysam.VariantFile(os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")))
        self.variant = Variant.from_pysam(record)

        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")
        self.params = argparse.Namespace()

    def test_resizing_image(self):
        image_tensor = images.create_single_example(self.params, self.variant, self.bam_path, "1:899722-900192")
        self.assertNotEqual(image_tensor.shape, (images.IMAGE_HEIGHT, 300, images.IMAGE_CHANNELS))

        image_tensor = images.create_single_example(
            self.params, self.variant, self.bam_path, "1:899722-900192", image_shape=(300, 300),
        )
        self.assertEqual(image_tensor.shape, (300, 300, images.IMAGE_CHANNELS))


def _mock_synthesize_variant_data(params, fasta_path, bam_path, allele_count, replicates):
    shutil.copyfile(os.path.join(FILE_DIR, "1_896922_902998.bam"), bam_path)
    shutil.copyfile(os.path.join(FILE_DIR, "1_896922_902998.bam.bai"), f"{bam_path}.bai")


class VCFExampleGenerateTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.params = argparse.Namespace(tempdir=self.tempdir.name, reference=None, flank=1, replicates=1)

        self.vcf_path = os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")
        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_vcf_generator_runs_without_error(self):
        all_examples = images.make_vcf_examples(self.params, self.vcf_path, self.bam_path)
        self.assertEqual(len(list(all_examples)), 1)

    def test_label_extraction(self):
        example = next(images.make_vcf_examples(self.params, self.vcf_path, self.bam_path))
        self.assertNotIn("label", example.features.feature)

        example = next(images.make_vcf_examples(self.params, self.vcf_path, self.bam_path, sample_or_label="HG002"))
        self.assertEqual(images._example_label(example), 2)

        example = next(images.make_vcf_examples(self.params, self.vcf_path, self.bam_path, sample_or_label=1))
        self.assertEqual(images._example_label(example), 1)

    def test_example_to_image(self):
        all_examples = images.make_vcf_examples(self.params, self.vcf_path, self.bam_path, image_shape=(300, 300))

        png_path = os.path.join(self.params.tempdir, "test.png")
        images.example_to_image(next(all_examples), png_path)

        self.assertTrue(os.path.exists(png_path))
        with Image.open(png_path) as image:
            self.assertEqual(image.size, (300, 300))

    def test_dataset_roundtrip(self):
        example = next(
            images.make_vcf_examples(
                self.params, self.vcf_path, self.bam_path, image_shape=(300, 300), sample_or_label="HG002",
            )
        )

        dataset_path = os.path.join(self.params.tempdir, "test.tfrecord")
        with tf.io.TFRecordWriter(dataset_path) as dataset:
            dataset.write(example.SerializeToString())
        self.assertTrue(os.path.exists(dataset_path))

        dataset = images.load_example_dataset(dataset_path, with_label=True)
        for features, label in dataset:
            self.assertEqual(features["image"].shape, (300, 300, images.IMAGE_CHANNELS))
            example_image = images._extract_image(example, (300, 300, images.IMAGE_CHANNELS))
            self.assertTrue(np.array_equal(features["image"], example_image))

            proto = images._features_variant(features)
            self.assertEqual(proto.svlen, -70)

            self.assertEqual(label, 2)

    @patch(
        "npsv2.variant._reference_sequence",
        return_value="GGCTGCGGGGAGGGGGGCGCGGGTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGATGTG",
    )
    @patch("npsv2.images._synthesize_variant_data", side_effect=_mock_synthesize_variant_data)
    def test_simulate_variant(self, synth_ref, mock_ref):
        example = next(
            images.make_vcf_examples(
                self.params,
                self.vcf_path,
                self.bam_path,
                image_shape=(300, 300),
                sample_or_label="HG002",
                simulate=True,
            )
        )
        self.assertEqual(mock_ref.call_count, 3)
        for args, _ in mock_ref.call_args_list:
            self.assertEqual(args[1], Range("1", 899921, 899993))

        self.assertIn("sim/replicates", example.features.feature)
        self.assertEqual(images._example_sim_replicates(example), 1)
        for ac in (0, 1, 2):
            self.assertIn(f"sim/{ac}/images/encoded", example.features.feature)

    @patch(
        "npsv2.variant._reference_sequence",
        return_value="GGCTGCGGGGAGGGGGGCGCGGGTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGATGTG",
    )
    @patch("npsv2.images._synthesize_variant_data", side_effect=_mock_synthesize_variant_data)
    def test_simulate_example_to_image(self, synth_ref, mock_ref):
        example = next(
            images.make_vcf_examples(
                self.params,
                self.vcf_path,
                self.bam_path,
                image_shape=(300, 300),
                sample_or_label="HG002",
                simulate=True,
            )
        )

        png_path = "test.png"  # os.path.join(self.params.tempdir, "test.png")
        images.example_to_image(example, png_path, with_simulations=True, margin=10, max_replicates=1)

        self.assertTrue(os.path.exists(png_path))
        with Image.open(png_path) as image:
            self.assertEqual(image.size, (920, 610))  # 3*300+2*10, 2*200+10

    @patch(
        "npsv2.variant._reference_sequence",
        return_value="GGCTGCGGGGAGGGGGGCGCGGGTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGATGTG",
    )
    @patch("npsv2.images._synthesize_variant_data", side_effect=_mock_synthesize_variant_data)
    def test_simulate_dataset_roundtrip(self, synth_ref, mock_ref):
        example = next(
            images.make_vcf_examples(
                self.params,
                self.vcf_path,
                self.bam_path,
                image_shape=(300, 300),
                sample_or_label="HG002",
                simulate=True,
            )
        )

        dataset_path = os.path.join(self.params.tempdir, "test.tfrecord")
        with tf.io.TFRecordWriter(dataset_path) as dataset:
            dataset.write(example.SerializeToString())
        self.assertTrue(os.path.exists(dataset_path))

        # Load dataset without simulated data
        dataset = images.load_example_dataset(dataset_path, with_label=True)
        for features, label in dataset:
            self.assertEqual(features["image"].shape, (300, 300, images.IMAGE_CHANNELS))
            self.assertEqual(label, 2)

        # Load dataset with simulated data
        dataset = images.load_example_dataset(dataset_path, with_label=True, with_simulations=True)
        for features, label in dataset:
            self.assertEqual(features["image"].shape, (300, 300, images.IMAGE_CHANNELS))
            self.assertEqual(label, 2)

            example_image = images._extract_image(example, (300, 300, images.IMAGE_CHANNELS))
            self.assertTrue(np.array_equal(features["image"], example_image))

            for ac in (0, 1, 2):
                feature_name = f"sim/{ac}/images"
                self.assertEqual(features[feature_name].shape, (1, 300, 300, images.IMAGE_CHANNELS))
                for repl in range(self.params.replicates):
                    self.assertTrue(np.array_equal(features[feature_name][repl], example_image))


class ChunkedVCFExampleGenerateTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.params = argparse.Namespace(tempdir=self.tempdir.name, reference=None, flank=1, replicates=1, threads=2)

        self.vcf_path = os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")
        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_region_generator(self):
        regions = list(images._region_generator(self.vcf_path))
        self.assertIn("1", regions)

    @patch(
        "npsv2.variant._reference_sequence",
        return_value="GGCTGCGGGGAGGGGGGCGCGGGTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGATGTG",
    )
    @patch("npsv2.images._synthesize_variant_data", side_effect=_mock_synthesize_variant_data)
    def test_make_dataset(self, synth_ref, mock_ref):
        dataset_path = os.path.join(self.params.tempdir, "test.tfrecord")
        images.vcf_to_tfrecords(
            self.params,
            self.vcf_path,
            self.bam_path,
            dataset_path,
            image_shape=(300, 300),
            sample_or_label="HG002",
            simulate=True,
        )

        self.assertEqual(mock_ref.call_count, 3)
        for args, _ in mock_ref.call_args_list:
            self.assertEqual(args[1], Range("1", 899921, 899993))

        self.assertTrue(os.path.exists(dataset_path))

        # Load dataset with simulated data
        dataset = images.load_example_dataset(dataset_path, with_label=True, with_simulations=True)
        for features, label in dataset:
            self.assertEqual(features["image"].shape, (300, 300, images.IMAGE_CHANNELS))
            self.assertEqual(label, 2)

            # png_path = "test.png" #os.path.join(self.params.tempdir, "test.png")
            # image = Image.fromarray(features["image"].numpy()[:,:,0], mode="L")
            # image.save(png_path)

            for ac in (0, 1, 2):
                feature_name = f"sim/{ac}/images"
                self.assertEqual(features[feature_name].shape, (1, 300, 300, images.IMAGE_CHANNELS))

    @patch(
        "npsv2.variant._reference_sequence",
        return_value="GGCTGCGGGGAGGGGGGCGCGGGTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGATGTG",
    )
    @patch("npsv2.images._synthesize_variant_data", side_effect=_mock_synthesize_variant_data)
    def test_compressed_dataset_roundtrip(self, synth_ref, mock_ref):
        dataset_path = os.path.join(self.params.tempdir, "test.tfrecord.gz")
        images.vcf_to_tfrecords(
            self.params, self.vcf_path, self.bam_path, dataset_path, image_shape=(300, 300), sample_or_label="HG002",
        )
        self.assertTrue(os.path.exists(dataset_path))
        # Load dataset with simulated data
        dataset = images.load_example_dataset(dataset_path, with_label=True)
        for features, label in dataset:
            self.assertEqual(features["image"].shape, (300, 300, images.IMAGE_CHANNELS))
            self.assertEqual(label, 2)
