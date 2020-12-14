import argparse, io, os, shutil, tempfile, unittest
from unittest.mock import patch, call
import pysam
import tensorflow as tf
import numpy as np
from PIL import Image
import ray
import warnings
from npsv2.variant import Variant
from npsv2.range import Range
from npsv2 import images
from npsv2 import npsv2_pb2
from npsv2.simulation import RandomVariants
from npsv2.sample import Sample

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")

def setUpModule():
    # Ignore resource warnings within Ray
    warnings.simplefilter("ignore", ResourceWarning)
    ray.init(num_cpus=1, num_gpus=0, local_mode=True, include_dashboard=False)

def tearDownModule():
    ray.shutdown()


def _mock_simulate_variant_sequencing(params, fasta_path, allele_count, sample: Sample, dir=tempfile.gettempdir()):
    return os.path.join(FILE_DIR, "1_896922_902998.bam")


def _mock_reference_sequence(reference_fasta, region):
    assert region.contig == "1"
    with pysam.FastaFile(os.path.join(FILE_DIR, "1_896922_902998.fasta")) as ref_fasta:
        return ref_fasta.fetch(reference=region.contig, start=region.start  - 896921, end=region.end  - 896921)


def _mock_chunk_genome(ref_path: str, vcf_path: str, chunk_size=30000000):
    return ["1:1-249250621"]


def _mock_generate_deletions(size, n=1, flank=0):
    with pysam.VariantFile(os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")) as vcf_file:
        for record in vcf_file:
            yield Variant.from_pysam(record)


class CreateSingleImageTest(unittest.TestCase):
    def setUp(self):
        record = next(pysam.VariantFile(os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")))
        self.variant = Variant.from_pysam(record)

        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")
        self.tempdir = tempfile.TemporaryDirectory()
        self.params = argparse.Namespace(
            reference=os.path.join(FILE_DIR, "1_896922_902998.fasta"),
            tempdir=self.tempdir.name,
            flank=1000,
            augment=False,
        )
        self.sample = Sample("HG002", mean_coverage=25.46, mean_insert_size=573.1, std_insert_size=164.2)

    def tearDown(self):
        self.tempdir.cleanup()

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_resizing_image(self, mock_ref):
        image_tensor = images.create_single_example(self.params, self.variant, self.bam_path, "1:899722-900192", self.sample)
        self.assertNotEqual(image_tensor.shape, (images.IMAGE_HEIGHT, 300, images.IMAGE_CHANNELS))

        image_tensor = images.create_single_example(
            self.params, self.variant, self.bam_path, "1:899722-900192", self.sample, image_shape=(300, 300),
        )
        self.assertEqual(image_tensor.shape, (300, 300, images.IMAGE_CHANNELS))


class VCFExampleGenerateTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.params = argparse.Namespace(
            tempdir=self.tempdir.name, reference=None, flank=1000, replicates=1, sample_ref=False, exclude_bed=None, augment=False,
        )

        self.sample = Sample("HG002", mean_coverage=25.46, mean_insert_size=573.1, std_insert_size=164.2)
        self.vcf_path = os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")
        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")

    def tearDown(self):
        self.tempdir.cleanup()

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_vcf_generator_runs_without_error(self, mock_ref):
        all_examples = images.make_vcf_examples(self.params, self.vcf_path, self.bam_path, self.sample)
        self.assertEqual(len(list(all_examples)), 1)

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_label_extraction(self, mock_ref):
        example = next(images.make_vcf_examples(self.params, self.vcf_path, self.bam_path, self.sample))
        self.assertNotIn("label", example.features.feature)

        example = next(images.make_vcf_examples(self.params, self.vcf_path, self.bam_path, self.sample, sample_or_label="HG002"))
        self.assertEqual(images._example_label(example), 2)

        example = next(images.make_vcf_examples(self.params, self.vcf_path, self.bam_path, self.sample, sample_or_label=1))
        self.assertEqual(images._example_label(example), 1)

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_example_to_image(self, mock_ref):
        all_examples = images.make_vcf_examples(self.params, self.vcf_path, self.bam_path, self.sample, image_shape=(300, 300))

        png_path = os.path.join(self.params.tempdir, "test.png")
        images.example_to_image(next(all_examples), png_path)

        self.assertTrue(os.path.exists(png_path))
        with Image.open(png_path) as image:
            self.assertEqual(image.size, (300, 300))

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_dataset_roundtrip(self, mock_ref):
        example = next(
            images.make_vcf_examples(
                self.params, self.vcf_path, self.bam_path, self.sample, image_shape=(300, 300), sample_or_label="HG002",
            )
        )

        dataset_path = os.path.join(self.params.tempdir, "test.tfrecord")
        with tf.io.TFRecordWriter(dataset_path) as dataset:
            dataset.write(example.SerializeToString())
        self.assertTrue(os.path.exists(dataset_path))

        dataset = images.load_example_dataset(dataset_path, with_label=True)
        for features, label in dataset:
            self.assertIn("image", features)
            self.assertEqual(features["image"].shape, (300, 300, images.IMAGE_CHANNELS))
            example_image = images._example_image(example)
            self.assertTrue(np.array_equal(features["image"], example_image))

            proto = images._features_variant(features)
            self.assertEqual(proto.svlen, -70)

            self.assertEqual(label, 2)

    # Since simulate_variant_sequencing is imported into images, we mock there...
    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    @patch("npsv2.images.simulate_variant_sequencing", side_effect=_mock_simulate_variant_sequencing)
    def test_simulate_variant(self, synth_ref, mock_ref):
        example = next(
            images.make_vcf_examples(
                self.params,
                self.vcf_path,
                self.bam_path,
                self.sample,
                image_shape=(300, 300),
                sample_or_label="HG002",
                simulate=True,
            )
        )
        self.assertEqual(mock_ref.call_count, 4)
        reference_query_region = Range("1", 899922, 899992).expand(self.params.flank)
        for args, _ in mock_ref.call_args_list:
            self.assertEqual(args[1], reference_query_region)
        
        self.assertIn("sim/images/shape", example.features.feature)
        self.assertEqual(images._example_sim_images_shape(example), (3, self.params.replicates, 300, 300, images.IMAGE_CHANNELS))

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    @patch("npsv2.images.simulate_variant_sequencing", side_effect=_mock_simulate_variant_sequencing)
    def test_simulate_example_to_image(self, synth_ref, mock_ref):
        example = next(
            images.make_vcf_examples(
                self.params,
                self.vcf_path,
                self.bam_path,
                self.sample,
                image_shape=(300, 300),
                sample_or_label="HG002",
                simulate=True,
            )
        )

        png_path = os.path.join(self.params.tempdir, "test.png")
        images.example_to_image(example, png_path, with_simulations=True, margin=10, max_replicates=1)

        self.assertTrue(os.path.exists(png_path))
        with Image.open(png_path) as image:
            self.assertEqual(image.size, (920, 610))  # 3*300+2*10, 2*200+10

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    @patch("npsv2.images.simulate_variant_sequencing", side_effect=_mock_simulate_variant_sequencing)
    def test_simulate_dataset_roundtrip(self, synth_ref, mock_ref):
        example = next(
            images.make_vcf_examples(
                self.params,
                self.vcf_path,
                self.bam_path,
                self.sample,
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

            example_image = images._example_image(example)
            self.assertTrue(np.array_equal(features["image"], example_image))

            self.assertIn("sim/images", features)
            sim_tensor = features["sim/images"]
            self.assertEqual(sim_tensor.shape, (3, self.params.replicates, 300, 300, images.IMAGE_CHANNELS))

            for ac in (0, 1, 2):
                for repl in range(self.params.replicates):
                    self.assertTrue(np.array_equal(sim_tensor[ac,repl], example_image))


    @patch("npsv2.simulation.RandomVariants._generate_deletions", side_effect=_mock_generate_deletions)
    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    @patch("npsv2.images.simulate_variant_sequencing", side_effect=_mock_simulate_variant_sequencing)
    def test_simulate_variant_with_sampled_ref(self, synth_ref, mock_ref, mock_rand_var):
        params = argparse.Namespace(**vars(self.params))
        setattr(params, "reference", os.path.join(FILE_DIR, "1_896922_902998.fasta"))
        setattr(params, "exclude_bed", os.path.join(FILE_DIR, "test_exclude.bed.gz"))
        setattr(params, "sim_ref", False)
        
        example = next(
            images.make_vcf_examples(
                params,
                self.vcf_path,
                self.bam_path,
                self.sample,
                image_shape=(300, 300),
                sample_or_label="HG002",
                simulate=True,
            )
        )
        self.assertEqual(mock_ref.call_count, 4)
        self.assertIn("sim/images/shape", example.features.feature)
        self.assertEqual(images._example_sim_images_shape(example), (3, self.params.replicates, 300, 300, images.IMAGE_CHANNELS))


class ChunkedVCFExampleGenerateTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.params = argparse.Namespace(
            tempdir=self.tempdir.name,
            reference=None,
            flank=1,
            replicates=1,
            threads=2,
            sample_ref=False,
            exclude_bed=None,
            augment=False,
        )
        self.sample = Sample("HG002", mean_coverage=25.46, mean_insert_size=573.1, std_insert_size=164.2)
        self.vcf_path = os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")
        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_region_dataset(self):
        regions = list(images._chunk_genome(os.path.join(FILE_DIR, "1_896922_902998.fasta"), self.vcf_path))
        self.assertIn("1:1-6067", regions)

    @patch("npsv2.images._chunk_genome", side_effect=_mock_chunk_genome)
    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    @patch("npsv2.images.simulate_variant_sequencing", side_effect=_mock_simulate_variant_sequencing)
    def test_make_dataset(self, synth_ref, mock_ref, mock_reg):
        dataset_path = os.path.join(self.params.tempdir, "test.tfrecord")
        images.vcf_to_tfrecords(
            self.params,
            self.vcf_path,
            self.bam_path,
            dataset_path,
            self.sample,
            image_shape=(300, 300),
            sample_or_label="HG002",
            simulate=True,
        )

        self.assertEqual(mock_ref.call_count, 4)
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

            self.assertEqual(features["sim/images"].shape, (3, self.params.replicates, 300, 300, images.IMAGE_CHANNELS))

    @patch("npsv2.images._chunk_genome", side_effect=_mock_chunk_genome)
    @patch(
        "npsv2.variant._reference_sequence",
        return_value="GGCTGCGGGGAGGGGGGCGCGGGTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGATGTG",
    )
    @patch("npsv2.images.simulate_variant_sequencing", side_effect=_mock_simulate_variant_sequencing)
    def test_compressed_dataset_roundtrip(self, synth_ref, mock_ref, mock_reg):
        dataset_path = os.path.join(self.params.tempdir, "test.tfrecord.gz")
        images.vcf_to_tfrecords(
            self.params, self.vcf_path, self.bam_path, dataset_path, self.sample, image_shape=(300, 300), sample_or_label="HG002",
        )
        self.assertTrue(os.path.exists(dataset_path))
        # Load dataset with simulated data
        dataset = images.load_example_dataset(dataset_path, with_label=True)
        for features, label in dataset:
            self.assertEqual(features["image"].shape, (300, 300, images.IMAGE_CHANNELS))
            self.assertEqual(label, 2)
