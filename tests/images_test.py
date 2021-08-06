import argparse, io, os, shutil, tempfile, unittest
from unittest.mock import patch, call
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
from npsv2 import npsv2_pb2
from npsv2.simulation import RandomVariants, bwa_index_loaded
from npsv2.sample import Sample

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")

def setUpModule():
    # Ignore resource warnings within Ray
    warnings.simplefilter("ignore", ResourceWarning)
    ray.init(num_cpus=1, num_gpus=0, local_mode=True, include_dashboard=False)

    hydra.initialize(config_path="../src/npsv2/conf")

def tearDownModule():
    ray.shutdown()
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def _mock_simulate_variant_sequencing(fasta_path, allele_count, sample: Sample, reference, shared_reference=None, dir=tempfile.gettempdir(), stats_path=None, gnomad_covg_path=None):
    return os.path.join(FILE_DIR, "1_896922_902998.bam")


def _mock_reference_sequence(reference_fasta, region, snv_vcf_path=None):
    assert region.contig == "1"
    with pysam.FastaFile(os.path.join(FILE_DIR, "1_896922_902998.fasta")) as ref_fasta:
        return ref_fasta.fetch(reference=region.contig, start=region.start-896921, end=region.end-896921)


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

@unittest.skip("Currently not in use")
class SingleHybridImageGeneratorClassTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(config_name="config", overrides=[
            "generator=single_hybrid",
            "pileup.image_width=200",
            "reference={}".format(os.path.join(FILE_DIR, "1_896922_902998.fasta")),
            "simulation.replicates=1",
        ])
        self.generator = hydra.utils.instantiate(self.cfg.generator, self.cfg)

        record = next(pysam.VariantFile(os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")))
        self.variant = Variant.from_pysam(record)
        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")
        self.sample = Sample("HG002", mean_coverage=25.46, mean_insert_size=573.1, std_insert_size=164.2)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_image_shape(self):
        self.assertEqual(self.generator.image_shape, (self.cfg.pileup.image_height, 200, 6))

    def test_region(self):
        image_region = self.generator.image_regions(self.variant)
        # Since the variant is less than the image width (and evenly sized), pad out to image width
        self.assertEqual(image_region.length, max(70 + 2*self.cfg.pileup.variant_padding, 200))

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_generate(self,  mock_ref):
        image_tensor = self.generator.generate(self.variant, self.bam_path, self.sample)
        self.assertEqual(image_tensor.shape, self.generator.image_shape)
        
        
        png_path = os.path.join(self.tempdir.name, "test.png")
        image = self.generator.render(image_tensor)
        image.save(png_path)
        self.assertTrue(os.path.exists(png_path))

    # Since simulate_variant_sequencing is imported into images, we mock there...
    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    @patch("npsv2.images.simulate_variant_sequencing", side_effect=_mock_simulate_variant_sequencing)
    def test_simulate_variant_to_example(self, synth_ref, mock_ref):
        example = images.make_variant_example(
            self.cfg,
            self.variant,
            self.bam_path,
            self.sample,
            simulate=True,
            generator=self.generator,
        )
        
        self.assertEqual(images._example_image_shape(example), self.generator.image_shape)
        self.assertEqual(images._example_sim_images_shape(example), (3, 1,) + self.generator.image_shape)
    
        png_path = os.path.join(self.tempdir.name, "test.png")
        images.example_to_image(self.cfg, example, png_path, with_simulations=True)
        self.assertTrue(os.path.exists(png_path))

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_generate_with_variant_strip(self,  mock_ref):
        # Reconfigure with variant band
        cfg = OmegaConf.merge(self.cfg, { "pileup": { "variant_band_height": 5 }})
        generator = hydra.utils.instantiate(self.cfg.generator, cfg)
        
        image_tensor = generator.generate(self.variant, self.bam_path, self.sample)
        self.assertEqual(image_tensor.shape, self.generator.image_shape)
        
        # The first 5 rows should all be identical
        for i in range(1,5):
            self.assertTrue(np.array_equal(image_tensor[i], image_tensor[0]))

        png_path = os.path.join(self.tempdir.name, "test.png")
        image = self.generator.render(image_tensor)
        image.save(png_path)
        self.assertTrue(os.path.exists(png_path))

@unittest.skip("Development only")
@unittest.skipUnless(os.path.exists("/data/human_g1k_v37.fasta") and bwa_index_loaded("/data/human_g1k_v37.fasta"), "Reference genome not available")
class SingleHybridImageGeneratorExampeTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(config_name="config", overrides=[
            "generator=single_hybrid",
            "reference=/data/human_g1k_v37.fasta",
            "shared_reference={}".format(os.path.basename('/data/human_g1k_v37.fasta')),
            "simulation.replicates=5",         
            "simulation.sample_ref=false",
        ])
        self.generator = hydra.utils.instantiate(self.cfg.generator, self.cfg)

        self.sample = Sample("HG002", mean_coverage=25.46, mean_insert_size=573.1, std_insert_size=164.2, sequencer="HS25", read_length=148)
        self.vcf_path = os.path.join(FILE_DIR, "12_22129565_22130387_DEL.vcf.gz")
        self.bam_path = os.path.join(FILE_DIR, "12_22127565_22132387.bam")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_normalized_allele_pixels(self):
        example = next(images.make_vcf_examples(self.cfg, self.vcf_path, self.bam_path, self.sample, simulate=True))
        png_path = os.path.join(self.tempdir.name, "test.png")
        images.example_to_image(self.cfg, example, png_path, with_simulations=True, max_replicates=5)

class SingleDepthImageGeneratorClassTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(config_name="config", overrides=[
            "generator=single_depth",
            "reference={}".format(os.path.join(FILE_DIR, "1_896922_902998.fasta")),
            "simulation.replicates=1",
        ])
        self.generator = hydra.utils.instantiate(self.cfg.generator, self.cfg)

        record = next(pysam.VariantFile(os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")))
        self.variant = Variant.from_pysam(record)
        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")
        self.sample = Sample("HG002", mean_coverage=25.46, mean_insert_size=573.1, std_insert_size=164.2, read_length=148)

    def tearDown(self):
        self.tempdir.cleanup()

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_generate(self,  mock_ref):
        image_tensor = self.generator.generate(self.variant, self.bam_path, self.sample)
        self.assertEqual(image_tensor.shape, self.generator.image_shape)
        
        png_path = os.path.join(self.tempdir.name, "test.png")
        image = self.generator.render(image_tensor)
        image.save(png_path)
        self.assertTrue(os.path.exists(png_path))

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_generate_with_variant_strip(self,  mock_ref):
        # Reconfigure with variant band
        cfg = OmegaConf.merge(self.cfg, { "pileup": { "variant_band_height": 5 }})
        generator = hydra.utils.instantiate(self.cfg.generator, cfg)
        
        image_tensor = generator.generate(self.variant, self.bam_path, self.sample)
        self.assertEqual(image_tensor.shape, self.generator.image_shape)
        
        # The first variant_band_height rows should all be identical
        for i in range(1,cfg.pileup.variant_band_height):
            self.assertTrue(np.array_equal(image_tensor[i], image_tensor[0]))

        png_path = "test.png" #os.path.join(self.tempdir.name, "test.png")
        image = self.generator.render(image_tensor)
        image.save(png_path)
        self.assertTrue(os.path.exists(png_path))

@unittest.skip("Development only")
@unittest.skipUnless(os.path.exists("/data/human_g1k_v37.fasta") and bwa_index_loaded("/data/human_g1k_v37.fasta"), "Reference genome not available")
class SingleDepthImageGeneratorExampeTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(config_name="config", overrides=[
            "generator=single_depth",
            "reference=/data/human_g1k_v37.fasta",
            "shared_reference={}".format(os.path.basename('/data/human_g1k_v37.fasta')),
            "simulation.replicates=2",         
            "simulation.sample_ref=false",
        ])
        self.generator = hydra.utils.instantiate(self.cfg.generator, self.cfg)

        self.sample = Sample("HG002", mean_coverage=25.46, mean_insert_size=573.1, std_insert_size=164.2, sequencer="HS25", read_length=148)
        self.vcf_path = os.path.join(FILE_DIR, "12_22129565_22130387_DEL.vcf.gz")
        self.bam_path = os.path.join(FILE_DIR, "12_22127565_22132387.bam")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_example_single_image(self):
        example = next(images.make_vcf_examples(self.cfg, self.vcf_path, self.bam_path, self.sample, simulate=True))
        png_path = os.path.join(self.tempdir.name, "test.png")
        images.example_to_image(self.cfg, example, png_path, with_simulations=True, max_replicates=2)

    def test_example_channel_image(self):
        example = next(images.make_vcf_examples(self.cfg, self.vcf_path, self.bam_path, self.sample, simulate=False))
        png_path = os.path.join(self.tempdir.name, "test.png")
        images.example_to_image(self.cfg, example, png_path, with_simulations=False, render_channels=True)

@unittest.skip("Currently not in use")
class SingleFragmentImageGeneratorClassTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(config_name="config", overrides=[
            "reference={}".format(os.path.join(FILE_DIR, "1_896922_902998.fasta")),
            "generator=single_fragment",
            "simulation.replicates=1",
            "pileup.image_width=1000",
            "pileup.image_height=100",
        ])
        self.generator = hydra.utils.instantiate(self.cfg.generator, self.cfg)

        record = next(pysam.VariantFile(os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")))
        self.variant = Variant.from_pysam(record)
        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")
        self.sample = Sample("HG002", mean_coverage=25.46, mean_insert_size=573.1, std_insert_size=164.2)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_image_shape(self):
        self.assertEqual(self.generator.image_shape, (100, 1000, 6))

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_generate(self,  mock_ref):
        image_tensor = self.generator.generate(self.variant, self.bam_path, self.sample)
        self.assertEqual(image_tensor.shape, self.generator.image_shape)
        
        png_path = os.path.join(self.tempdir.name, "test.png")
        image = self.generator.render(image_tensor)
        image.save(png_path)
        self.assertTrue(os.path.exists(png_path))

@unittest.skip("Currently not in use")
class WindowedReadImageGeneratorClassTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(config_name="config", overrides=[
            "reference={}".format(os.path.join(FILE_DIR, "1_896922_902998.fasta")),
            "generator=windowed_read",
            "simulation.replicates=1",
        ])
        self.generator = hydra.utils.instantiate(self.cfg.generator, self.cfg)

        record = next(pysam.VariantFile(os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")))
        self.variant = Variant.from_pysam(record)
        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")
        self.sample = Sample("HG002", mean_coverage=25.46, mean_insert_size=573.1, std_insert_size=164.2)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_image_shape(self):
        self.assertEqual(self.generator.image_shape, (None, self.cfg.pileup.image_height, 50, 6))

    def test_region(self):
        image_regions = self.generator.image_regions(self.variant)
        self.assertEqual(len(image_regions), 5)

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_generate(self,  mock_ref):
        image_tensor = self.generator.generate(self.variant, self.bam_path, self.sample)
        # For this size variant (with 1 flank window) there should be 5 windows
        self.assertEqual(image_tensor.shape, (5,) + self.generator.image_shape[1:])
        
        png_path = os.path.join(self.tempdir.name, "test.png")
        image = self.generator.render(image_tensor)
        image.save(png_path)
        self.assertTrue(os.path.exists(png_path))
    

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    @patch("npsv2.images.simulate_variant_sequencing", side_effect=_mock_simulate_variant_sequencing)
    def test_simulate_variant_to_example(self, synth_ref, mock_ref):
        example = images.make_variant_example(
            self.cfg,
            self.variant,
            self.bam_path,
            self.sample,
            simulate=True,
            generator=self.generator,
        )
        
        self.assertEqual(images._example_image_shape(example), (5,) + self.generator.image_shape[1:])
        self.assertEqual(images._example_sim_images_shape(example), (3, 1, 5) + self.generator.image_shape[1:])
    
        png_path = os.path.join(self.tempdir.name, "test.png")
        images.example_to_image(self.cfg, example, png_path, with_simulations=True)
        self.assertTrue(os.path.exists(png_path))

@unittest.skip("Currently not in use")
class BreakpointReadImageGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(config_name="config", overrides=[
            "reference={}".format(os.path.join(FILE_DIR, "1_896922_902998.fasta")),
            "generator=breakpoint_read",
            "simulation.replicates=1",
        ])
        self.generator = hydra.utils.instantiate(self.cfg.generator, self.cfg)

        record = next(pysam.VariantFile(os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")))
        self.variant = Variant.from_pysam(record)
        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")
        self.sample = Sample("HG002", mean_coverage=25.46, mean_insert_size=573.1, std_insert_size=164.2)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_image_shape(self):
        self.assertEqual(self.generator.image_shape, (2, self.cfg.pileup.image_height, self.cfg.pileup.image_width, 6))

    def test_region(self):
        image_regions = self.generator.image_regions(self.variant)
        self.assertEqual(len(image_regions), 2)

    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_generate(self,  mock_ref):
        image_tensor = self.generator.generate(self.variant, self.bam_path, self.sample)
        self.assertEqual(image_tensor.shape, self.generator.image_shape)
        
        png_path =os.path.join(self.tempdir.name, "test.png")
        image = self.generator.render(image_tensor)
        image.save(png_path)
        self.assertTrue(os.path.exists(png_path))


class VCFExampleGenerateTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(config_name="config", overrides=[
            "reference=placeholder.fasta",
            "simulation.replicates=1",
            "pileup.render_snv=false",
        ])
        self.generator = hydra.utils.instantiate(self.cfg.generator, self.cfg)

        self.sample = Sample("HG002", mean_coverage=25.46, mean_insert_size=573.1, std_insert_size=164.2, read_length=148)
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

        example = next(images.make_vcf_examples(self.cfg, self.vcf_path, self.bam_path, self.sample, sample_or_label="HG002"))
        self.assertEqual(images._example_label(example), 2)

        example = next(images.make_vcf_examples(self.cfg, self.vcf_path, self.bam_path, self.sample, sample_or_label=1))
        self.assertEqual(images._example_label(example), 1)


    @patch("npsv2.variant._reference_sequence", side_effect=_mock_reference_sequence)
    def test_dataset_roundtrip(self, mock_ref):
        example = next(
            images.make_vcf_examples(
                self.cfg, self.vcf_path, self.bam_path, self.sample, sample_or_label="HG002",
            )
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
        cfg = OmegaConf.merge(self.cfg, { "pileup": { "realigner_flank": 1 }})
        dataset_path = os.path.join(self.tempdir.name, "test.tfrecords.gz")
        images.vcf_to_tfrecords(
            cfg,
            self.vcf_path,
            self.bam_path,
            dataset_path,
            self.sample,
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
            self.assertEqual(features["image"].shape, self.generator.image_shape)
            self.assertEqual(label, 2)
            self.assertEqual(features["sim/images"].shape, (3, 1) + self.generator.image_shape)

            png_path = os.path.join(self.tempdir.name, "test.png")
            images.features_to_image(self.cfg, features, png_path, with_simulations=True)
            self.assertTrue(os.path.exists(png_path))

@unittest.skipUnless(os.path.exists("/data/human_g1k_v37.fasta") and bwa_index_loaded("/data/human_g1k_v37.fasta"), "Reference genome not available")
class MultiallelicVCFExampleGenerateTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(config_name="config", overrides=[
            "reference=/data/human_g1k_v37.fasta",
            "shared_reference={}".format(os.path.basename('/data/human_g1k_v37.fasta')),
            "generator=single_depth",
            "simulation.replicates=1",
            "pileup.render_snv=false",
        ])
        self.generator = hydra.utils.instantiate(self.cfg.generator, self.cfg)

        self.sample = Sample("HG002", mean_coverage=25.46, mean_insert_size=573.1, std_insert_size=164.2, sequencer="HS25", read_length=148)
        self.vcf_path = os.path.join(FILE_DIR, "11_647806_647946_DEL.vcf.gz")
        self.bam_path = os.path.join(FILE_DIR, "11_645806_649946.bam")

    def tearDown(self):
        self.tempdir.cleanup()
    
    def test_multiallelic_variant(self):
        record = next(pysam.VariantFile(self.vcf_path))
        variant = Variant.from_pysam(record)

        example = images.make_variant_example(
            self.cfg,
            variant,
            self.bam_path,
            self.sample,
            simulate=True,
            generator=self.generator,
            alleles={1, 2}
        )

        png_path = os.path.join(self.tempdir.name, "test.png")
        images.example_to_image(self.cfg, example, png_path, with_simulations=True)

@unittest.skipUnless(os.path.exists("/data/human_g1k_v37.fasta") and bwa_index_loaded("/data/human_g1k_v37.fasta"), "Reference genome not available")
class SNVRenderTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(config_name="config", overrides=[
            "reference=/data/human_g1k_v37.fasta",
            "shared_reference={}".format(os.path.basename('/data/human_g1k_v37.fasta')),
            "generator=single_depth",
            "simulation.replicates=1",
            "pileup.render_snv=true",
            "pileup.snv_vcf_input={}".format(os.path.join(FILE_DIR, "1_67806460_67811624.snvs.vcf.gz")),
        ])

        self.sample = Sample("HG002", mean_coverage=25.46, mean_insert_size=573.1, std_insert_size=164.2, sequencer="HS25", read_length=148)
        self.vcf_path = os.path.join(FILE_DIR, "1_67808460_67808624_DEL.vcf.gz")
        self.bam_path = os.path.join(FILE_DIR, "1_67806460_67811624.bam")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_normalized_allele_pixels(self):
        example = next(images.make_vcf_examples(self.cfg, self.vcf_path, self.bam_path, self.sample, simulate=True))
        png_path = os.path.join(self.tempdir.name, "test.png")
        images.example_to_image(self.cfg, example, png_path, with_simulations=True)