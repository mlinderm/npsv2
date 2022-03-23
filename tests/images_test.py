import os, tempfile, unittest
from unittest.mock import patch, call
from parameterized import parameterized, parameterized_class
import pysam
import tensorflow as tf
import numpy as np
from PIL import Image
import ray
import warnings
import hydra
from omegaconf import OmegaConf
from npsv2.variant import Variant
from npsv2.range import Range
from npsv2 import images
from npsv2.simulation import bwa_index_loaded
from npsv2.sample import Sample

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")


def setUpModule():
    # Ignore resource warnings within Ray
    warnings.simplefilter("ignore", ResourceWarning)
    ray.init(num_cpus=1, num_gpus=0, local_mode=True, include_dashboard=False)

    hydra.initialize(config_path="../src/npsv2/conf")


def tearDownModule():
    ray.shutdown()
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def _mock_simulate_variant_sequencing(
    fasta_path,
    allele_count,
    sample: Sample,
    reference,
    shared_reference=None,
    dir=tempfile.gettempdir(),
    stats_path=None,
    gnomad_covg_path=None,
):
    return os.path.join(FILE_DIR, "1_896922_902998.bam")


def _mock_reference_sequence(reference_fasta, region, snv_vcf_path=None):
    assert region.contig == "1"
    with pysam.FastaFile(os.path.join(FILE_DIR, "1_896922_902998.fasta")) as ref_fasta:
        return ref_fasta.fetch(reference=region.contig, start=region.start - 896921, end=region.end - 896921)


class ImageGeneratorConfigTest(unittest.TestCase):
    def test_inferred_config(self):
        cfg = hydra.compose(config_name="config", overrides=[])
        self.assertEqual(cfg.pileup.aligned_base_pixel, cfg.pileup.match_base_pixel)

    def test_instantiate_generator(self):
        cfg = hydra.compose(config_name="config", overrides=[])
        generator = hydra.utils.instantiate(cfg.generator, cfg)
        self.assertIsInstance(generator, images.SingleDepthImageGenerator)

    def test_override_generator(self):
        cfg = hydra.compose(config_name="config", overrides=["generator=windowed_read"])
        generator = hydra.utils.instantiate(cfg.generator, cfg)
        self.assertIsInstance(generator, images.WindowedReadImageGenerator)
        self.assertIn("flank_windows", cfg.pileup)


class ImageRegionTest(unittest.TestCase):
    def setUp(self):
        self.cfg = hydra.compose(
            config_name="config",
            overrides=[
               "pileup.image_width=300",
               "pileup.variant_padding=100",
            ],
        )

    def test_small(self):
        self.assertEqual(images.image_region(self.cfg, Range("1",900297,900298)).length, 300)
        self.assertEqual(images.image_region(self.cfg, Range("1",900296,900298)).length, 300)

    def test_large(self):
        self.assertGreater(images.image_region(self.cfg, Range("12",22129564,22130387)).length, 300)


class SingleDepthImageGeneratorClassTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(
            config_name="config",
            overrides=[
                "generator=single_depth",
                "reference={}".format(os.path.join(FILE_DIR, "1_896922_902998.fasta")),
                "simulation.replicates=1",
            ],
        )
        self.generator = hydra.utils.instantiate(self.cfg.generator, self.cfg)

        record = next(pysam.VariantFile(os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")))
        self.variant = Variant.from_pysam(record)
        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")
        self.sample = Sample(
            "HG002",
            mean_coverage=25.46,
            mean_insert_size=573.1,
            std_insert_size=164.2,
            read_length=148,
            chrom_normalized_coverage={"1": 1.5},
        )

    def tearDown(self):
        self.tempdir.cleanup()

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_generate(self, mock_ref):
        image_tensor = self.generator.generate(self.variant, self.bam_path, self.sample)
        self.assertEqual(image_tensor.shape, self.generator.image_shape)

        png_path = os.path.join(self.tempdir.name, "test.png")
        image = self.generator.render(image_tensor)
        image.save(png_path)
        self.assertTrue(os.path.exists(png_path))

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_generate_with_variant_strip(self, mock_ref):
        # Reconfigure with variant band
        cfg = OmegaConf.merge(self.cfg, {"pileup": {"variant_band_height": 5}})
        generator = hydra.utils.instantiate(self.cfg.generator, cfg)

        image_tensor = generator.generate(self.variant, self.bam_path, self.sample)
        self.assertEqual(image_tensor.shape, self.generator.image_shape)

        # The first variant_band_height rows should all be identical
        for i in range(1, cfg.pileup.variant_band_height):
            self.assertTrue(np.array_equal(image_tensor[i], image_tensor[0]))

        png_path = "test.png"  # os.path.join(self.tempdir.name, "test.png")
        image = self.generator.render(image_tensor)
        image.save(png_path)
        self.assertTrue(os.path.exists(png_path))

    # Since _reference_sequence is imported into images we mock there too
    @patch("npsv2.images._reference_sequence", side_effect=_mock_reference_sequence)
    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    @patch("npsv2.images.simulate_variant_sequencing", side_effect=_mock_simulate_variant_sequencing)
    def test_simulate_chrom_coverage(self, mock_sim, mock_var_ref, mock_images_ref):
        self.assertAlmostEqual(self.sample.chrom_mean_coverage("1"), 1.5 * self.sample.mean_coverage, places=2)

        # Reconfigure chrom_norm_covg
        cfg = OmegaConf.merge(self.cfg, {"simulation": {"chrom_norm_covg": True}})
        print(self.sample.chrom_mean_coverage("1"))
        images.make_variant_example(
            cfg, self.variant, self.bam_path, self.sample, simulate=True, generator=self.generator,
        )
        self.assertEqual(mock_sim.call_count, 3, msg="Should be called for each genotype")
        hap_coverages = [call[0][1] for call in mock_sim.call_args_list]
        np.testing.assert_allclose(hap_coverages, np.array([1.0, 0.5, 1.0]) * 1.5 * self.sample.mean_coverage)


#@unittest.skip("Development only")
@unittest.skipUnless(
    os.path.exists("/data/human_g1k_v37.fasta")
    and bwa_index_loaded("/data/human_g1k_v37.fasta")
    and os.path.exists("/data/HG002-ready.bam"),
    "Reference genome or HG002 sequencing data not available",
)
@parameterized_class(
    [
        # { "vcf_path": os.path.join(FILE_DIR, "12_22129565_22130387_DEL.vcf.gz") }, # Presentation example
        # { "vcf_path": os.path.join(FILE_DIR, "2_1521325_1521397_DEL.vcf.gz") }, # Undercall
        # { "vcf_path": os.path.join(FILE_DIR, "21_46906303_46906372_DEL.vcf.gz") }, # Overcall
        # { "vcf_path": os.path.join(FILE_DIR, "4_898004_898094_DEL.vcf.gz") }, # Overcall
        # {"vcf_path": os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")},  # Offset (GIAB)
        # {"vcf_path": os.path.join(FILE_DIR, "1_900011_900086_DEL.vcf.gz")},  # Offset (PBSV)
        {"vcf_path": os.path.join(FILE_DIR, "1_1865644_1866241_DEL.vcf")},  # Offset (GIAB)
        {"vcf_path": os.path.join(FILE_DIR, "1_1866394_1867006_DEL.vcf")},  # Offset (Proposal)
        {"vcf_path": os.path.join(FILE_DIR, "1_1866396_1867023_DEL.vcf")},  # Offset (PBSV)
        # { "vcf_path": os.path.join(FILE_DIR, "5_126180130_126180259_DEL.vcf") }, # Haplotagged
        # {"vcf_path": os.path.join(FILE_DIR, "5_126180060_126180189_DEL.vcf")},  # Haplotagged

    ]
)
class SingleDepthImageGeneratorExampeTest(unittest.TestCase):
    """Generate example images for presentations, etc. Requires reference genome, b37 HG002 BAM, etc."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(
            config_name="config",
            overrides=[
                "generator=single_depth",
                "reference=/data/human_g1k_v37.fasta",
                "shared_reference={}".format(os.path.basename("/data/human_g1k_v37.fasta")),
                "simulation.replicates=1",
                "simulation.sample_ref=false",
                f"simulation.save_sim_bam_dir={RESULT_DIR}",
            ],
        )
        self.generator = hydra.utils.instantiate(self.cfg.generator, self.cfg)
        self.sample = Sample(
            "HG002",
            mean_coverage=25.46,
            mean_insert_size=573.1,
            std_insert_size=164.2,
            sequencer="HS25",
            read_length=148,
        )
        self.bam_path = "/data/HG002-ready.bam"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_example_single_image(self):
        example = next(images.make_vcf_examples(self.cfg, self.vcf_path, self.bam_path, self.sample, simulate=True))
        png_path = os.path.join(RESULT_DIR, os.path.splitext(os.path.basename(self.vcf_path))[0] + ".png")
        images.example_to_image(self.cfg, example, png_path, with_simulations=True, max_replicates=2)

    def test_example_channel_image(self):
        example = next(images.make_vcf_examples(self.cfg, self.vcf_path, self.bam_path, self.sample, simulate=True))
        png_path = os.path.join(RESULT_DIR, os.path.splitext(os.path.basename(self.vcf_path))[0] + ".channel.png")
        images.example_to_image(
            self.cfg, example, png_path, with_simulations=True, max_replicates=1, render_channels=True
        )


class VCFExampleGenerateTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(
            config_name="config",
            overrides=["reference=placeholder.fasta", "simulation.replicates=1", "pileup.render_snv=false",],
        )
        self.generator = hydra.utils.instantiate(self.cfg.generator, self.cfg)

        self.sample = Sample(
            "HG002", mean_coverage=25.46, mean_insert_size=573.1, std_insert_size=164.2, read_length=148
        )
        self.vcf_path = os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")
        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")

    def tearDown(self):
        self.tempdir.cleanup()

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_vcf_generator_runs_without_error(self, mock_ref):
        all_examples = images.make_vcf_examples(self.cfg, self.vcf_path, self.bam_path, self.sample)
        self.assertEqual(len(list(all_examples)), 1)

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_label_extraction(self, mock_ref):
        example = next(images.make_vcf_examples(self.cfg, self.vcf_path, self.bam_path, self.sample))
        self.assertNotIn("label", example.features.feature)

        example = next(
            images.make_vcf_examples(self.cfg, self.vcf_path, self.bam_path, self.sample, sample_or_label="HG002")
        )
        self.assertEqual(images._example_label(example), 2)

        example = next(
            images.make_vcf_examples(self.cfg, self.vcf_path, self.bam_path, self.sample, sample_or_label=1)
        )
        self.assertEqual(images._example_label(example), 1)

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_dataset_roundtrip(self, mock_ref):
        example = next(
            images.make_vcf_examples(self.cfg, self.vcf_path, self.bam_path, self.sample, sample_or_label="HG002",)
        )

        dataset_path = os.path.join(self.tempdir.name, "test.tfrecord")
        with tf.io.TFRecordWriter(dataset_path) as dataset:
            dataset.write(example.SerializeToString())
        self.assertTrue(os.path.exists(dataset_path))

        dataset = images.load_example_dataset(dataset_path, with_label=True)
        for features, label in dataset:
            self.assertIn("image", features)
            self.assertEqual(features["image"].shape, self.generator.image_shape)

            example_image = images._example_image(example)
            self.assertTrue(np.array_equal(features["image"], example_image))

            proto = images._features_variant(features)
            self.assertEqual(proto.svlen, [-70])

            self.assertEqual(label, 2)

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    @patch("npsv2.images.simulate_variant_sequencing", side_effect=_mock_simulate_variant_sequencing)
    def test_vcf_to_tfrecords(self, synth_ref, mock_ref):
        cfg = OmegaConf.merge(self.cfg, {"pileup": {"realigner_flank": 1}})
        dataset_path = os.path.join(self.tempdir.name, "test.tfrecords.gz")
        images.vcf_to_tfrecords(
            cfg, self.vcf_path, self.bam_path, dataset_path, self.sample, sample_or_label="HG002", simulate=True,
        )

        self.assertEqual(mock_ref.call_count, 4)
        for args, _ in mock_ref.call_args_list:
            self.assertEqual(args[1], Range("1", 899921, 899993))

        self.assertTrue(os.path.exists(dataset_path))

        # Load dataset with simulated data
        dataset = images.load_example_dataset(dataset_path, with_label=True, with_simulations=True)
        for features, label in dataset:
            self.assertEqual(features["image"].shape, self.generator.image_shape)
            self.assertEqual(label, 2)
            self.assertEqual(features["sim/images"].shape, (3, 1) + self.generator.image_shape)

            png_path = os.path.join(self.tempdir.name, "test.png")
            images.features_to_image(self.cfg, features, png_path, with_simulations=True)
            self.assertTrue(os.path.exists(png_path))


@unittest.skipUnless(
    os.path.exists("/data/human_g1k_v37.fasta") and bwa_index_loaded("/data/human_g1k_v37.fasta"),
    "Reference genome not available",
)
class MultiallelicVCFExampleGenerateTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(
            config_name="config",
            overrides=[
                "reference=/data/human_g1k_v37.fasta",
                "shared_reference={}".format(os.path.basename("/data/human_g1k_v37.fasta")),
                "generator=single_depth",
                "simulation.replicates=1",
            ],
        )
        self.generator = hydra.utils.instantiate(self.cfg.generator, self.cfg)

        self.sample = Sample(
            "HG002",
            mean_coverage=25.46,
            mean_insert_size=573.1,
            std_insert_size=164.2,
            sequencer="HS25",
            read_length=148,
        )
        self.vcf_path = os.path.join(FILE_DIR, "11_647806_647946_DEL.vcf.gz")
        self.bam_path = os.path.join(FILE_DIR, "11_645806_649946.bam")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_multiallelic_variant(self):
        record = next(pysam.VariantFile(self.vcf_path))
        variant = Variant.from_pysam(record)

        example = images.make_variant_example(
            self.cfg, variant, self.bam_path, self.sample, simulate=True, generator=self.generator, alleles={1, 2}
        )

        png_path = os.path.join(self.tempdir.name, "test.png")
        images.example_to_image(self.cfg, example, png_path, with_simulations=True)


@unittest.skipUnless(
    os.path.exists("/data/human_g1k_v37.fasta") and bwa_index_loaded("/data/human_g1k_v37.fasta"),
    "Reference genome not available",
)
class SNVRenderTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(
            config_name="config",
            overrides=[
                "reference=/data/human_g1k_v37.fasta",
                "shared_reference={}".format(os.path.basename("/data/human_g1k_v37.fasta")),
                "generator=single_depth",
                "simulation.replicates=1",
                "pileup.render_snv=true",
                "pileup.snv_vcf_input={}".format(os.path.join(FILE_DIR, "1_67806460_67811624.snvs.vcf.gz")),
            ],
        )

        self.sample = Sample(
            "HG002",
            mean_coverage=25.46,
            mean_insert_size=573.1,
            std_insert_size=164.2,
            sequencer="HS25",
            read_length=148,
        )
        self.vcf_path = os.path.join(FILE_DIR, "1_67808460_67808624_DEL.vcf.gz")
        self.bam_path = os.path.join(FILE_DIR, "1_67806460_67811624.bam")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_normalized_allele_pixels(self):
        example = next(images.make_vcf_examples(self.cfg, self.vcf_path, self.bam_path, self.sample, simulate=True))
        png_path = os.path.join(self.tempdir.name, "test.png")
        images.example_to_image(self.cfg, example, png_path, with_simulations=True)


@unittest.skipUnless(
    os.path.exists("/data/human_g1k_v37.fasta") and bwa_index_loaded("/data/human_g1k_v37.fasta"),
    "Reference genome not available",
)
@parameterized_class([
    {"vcf_path": os.path.join(FILE_DIR, "1_931634_931634_INS.vcf.gz"), "bam_path": os.path.join(FILE_DIR, "1_931634_931634.bam")},
    {"vcf_path": os.path.join(FILE_DIR, "1_900298_900298_SUB.vcf.gz"), "bam_path": os.path.join(FILE_DIR, "1_896922_902998.bam")},
])
class KindSingleDepthImageGeneratorClassTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(
            config_name="config",
            overrides=[
                "generator=single_depth",
                "reference=/data/human_g1k_v37.fasta",
                "simulation.replicates=1",
            ],
        )
        self.generator = hydra.utils.instantiate(self.cfg.generator, self.cfg)

        record = next(pysam.VariantFile(self.vcf_path))
        self.variant = Variant.from_pysam(record)
        self.bam_path = self.bam_path
        self.sample = Sample(
            "HG002",
            mean_coverage=25.46,
            mean_insert_size=573.1,
            std_insert_size=164.2,
            sequencer="HS25",
            read_length=148,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_generate(self):
        image_tensor = self.generator.generate(self.variant, self.bam_path, self.sample)
        self.assertEqual(image_tensor.shape, self.generator.image_shape)

        png_path = os.path.join(self.tempdir.name, "test.png")
        image = self.generator.render(image_tensor)
        image.save(png_path)
        self.assertTrue(os.path.exists(png_path))

    def test_single_image(self):
        example = images.make_variant_example(
            self.cfg, self.variant, self.bam_path, self.sample, simulate=True, generator=self.generator,
        )
        png_path = os.path.join(self.tempdir.name, "test.png")
        images.example_to_image(self.cfg, example, png_path, with_simulations=True, max_replicates=1)
        self.assertTrue(os.path.exists(png_path))