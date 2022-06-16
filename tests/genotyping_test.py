import os, tempfile, unittest
from unittest.mock import patch, call
import pysam
import ray
import warnings
import hydra
import tensorflow as tf
import numpy as np
from npsv2.variant import Variant
from npsv2.sample import Sample
from npsv2 import genotyping
from npsv2.simulation import bwa_index_loaded

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")


def setUpModule():
    # Ignore resource warnings within Ray
    warnings.simplefilter("ignore", ResourceWarning)
    ray.init(num_cpus=1, num_gpus=0, local_mode=True, include_dashboard=False)

    hydra.initialize(config_path="../src/npsv2/conf")


def tearDownModule():
    ray.shutdown()
    hydra.core.global_hydra.GlobalHydra.instance().clear()


@unittest.skipUnless(
    os.path.exists("/data/human_g1k_v37.fasta") and bwa_index_loaded("/data/human_g1k_v37.fasta"),
    "Reference genome not available",
)
class MultiallelicVCFGenotypeTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(
            config_name="config",
            overrides=[
                "reference=/data/human_g1k_v37.fasta",
                "shared_reference={}".format(os.path.basename("/data/human_g1k_v37.fasta")),
                "model=joint_embeddings",
                "simulation.replicates=1",
            ],
        )
        self.generator = hydra.utils.instantiate(self.cfg.generator, self.cfg)

        self.vcf_path = os.path.join(FILE_DIR, "11_647806_647946_DEL.vcf.gz")
        self.bam_path = os.path.join(FILE_DIR, "11_645806_649946.bam")
        self.sample = Sample(
            "HG002",
            mean_coverage=25.46,
            mean_insert_size=573.1,
            std_insert_size=164.2,
            sequencer="HS25",
            read_length=148,
        )
        self.sample.bam = self.bam_path

    def tearDown(self):
        self.tempdir.cleanup()

    def test_allele_masks(self):
        record = next(pysam.VariantFile(self.vcf_path))
        variant = Variant.from_pysam(record)
        self.assertEqual(genotyping._allele_masks(variant), [{1}, {2}, {1, 2}])

    @patch("npsv2.models.JointEmbeddingsModel")
    def test_genotype_multiallelic(self, mock_model):
        # Mock distances returned by model predict function
        instance = mock_model.return_value
        instance.predict.side_effect = [
            (None, tf.constant([[0.5, 0.5, 0.9]]), None, None),
            (None, tf.constant([[0.51, 0.51, 0.92]]), None, None),
            (None, tf.constant([[0.52, 0.52, 0.12]]), None, None),
        ]

        vcf_path = os.path.join(self.tempdir.name, "test.vcf")
        genotyping.genotype_vcf(self.cfg, self.vcf_path, {"HG002": self.sample}, ["null.h5"], vcf_path)
        self.assertEqual(instance.predict.call_count, 3)

        record = next(pysam.VariantFile(vcf_path))
        self.assertEqual(record.samples["HG002"].allele_indices, (1, 2))
        # Distances follow VCF genotype likelihood ordering
        np.testing.assert_array_almost_equal(record.samples["HG002"]["DS"], (0.51, 0.5, 0.9, 0.51, 0.12, 0.92))
        np.testing.assert_array_almost_equal(record.samples["HG002"]["DHFFC"], (0.27937498688697815, 0.3624579906463623))


@unittest.skipUnless(
    os.path.exists("/data/human_g1k_v37.fasta")
    and bwa_index_loaded("/data/human_g1k_v37.fasta")
    and os.path.exists("/data/HG002-ready.bam"),
    "Reference genome or HG002 sequencing data not available",
)
class EmbeddingsOutputTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.embeddings_path = os.path.join("tests", "results", "test.tfrecords.gz")
        self.cfg = hydra.compose(
            config_name="config",
            overrides=[
                "reference=/data/human_g1k_v37.fasta",
                "shared_reference={}".format(os.path.basename("/data/human_g1k_v37.fasta")),
                "model=joint_embeddings",
                "simulation.replicates=1",
                f"embeddings_output={self.embeddings_path}"
            ],
        )
        self.generator = hydra.utils.instantiate(self.cfg.generator, self.cfg)

        self.vcf_path = os.path.join(FILE_DIR, "refine_input.vcf")
        self.bam_path = "/data/HG002-ready.bam"
        self.sample = Sample(
            "HG002",
            mean_coverage=25.46,
            mean_insert_size=573.1,
            std_insert_size=164.2,
            sequencer="HS25",
            read_length=148,
        )
        self.sample.bam = self.bam_path

    def tearDown(self):
        self.tempdir.cleanup()

    @patch("npsv2.models.JointEmbeddingsModel")
    def test_embeddings_file(self, mock_model):
        # Mock distances returned by model predict function
        instance = mock_model.return_value
        instance.predict.return_value = (None, tf.constant([[0.5, 0.5, 0.9]]), tf.ones((1, 512)), tf.ones((1, 3, 512)))

        output_vcf_path = os.path.join(self.tempdir.name, "test.vcf")
        genotyping.genotype_vcf(self.cfg, self.vcf_path, {"HG002": self.sample}, ["null.h5"], output_vcf_path)
        self.assertTrue(os.path.exists(self.embeddings_path))

        dataset = genotyping.load_embeddings_dataset(self.embeddings_path)
        for features in dataset:
            print(features)
            self.assertIn("support_embeddings", features)
            self.assertIn("query_embeddings", features)