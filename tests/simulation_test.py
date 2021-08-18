import argparse, io, os, sys, tempfile, unittest
from unittest.mock import patch
import pysam
import hydra
from omegaconf import OmegaConf
from npsv2.variant import Variant
from npsv2.sample import Sample
from npsv2.simulation import RandomVariants, simulate_variant_sequencing, bwa_index_loaded

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")


def setUpModule():
    hydra.initialize(config_path="../src/npsv2/conf")


def tearDownModule():
    hydra.core.global_hydra.GlobalHydra.instance().clear()


class GenerateRandomDeletions(unittest.TestCase):
    def setUp(self):
        self.generator = RandomVariants(
            os.path.join(FILE_DIR, "1_896922_902998.fasta"), os.path.join(FILE_DIR, "test_exclude.bed.gz"),
        )

    def test_deletion_generator(self):
        variants = list(self.generator._generate_deletions(100, n=2))
        self.assertEqual(len(variants), 2)

        for variant in variants:
            self.assertEqual(variant.length_change(), -100)


@unittest.skipUnless(
    os.path.exists("/data/human_g1k_v37.fasta") and bwa_index_loaded("/data/human_g1k_v37.fasta"),
    "Reference genome not available",
)
class NormalizeCoverage(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = hydra.compose(
            config_name="config",
            overrides=[
                "generator=single_depth",
                "reference=/data/human_g1k_v37.fasta",
                "shared_reference=human_g1k_v37.fasta",
            ],
        )

        record = next(pysam.VariantFile(os.path.join(FILE_DIR, "1_900011_900086_DEL.vcf.gz")))
        self.variant = Variant.from_pysam(record)
        self.bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")
        self.sample = Sample("HG002", mean_coverage=25.46, mean_insert_size=573.1, std_insert_size=164.2, sequencer="HS25", read_length=148)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_gc_normalized_coverage(self):
        fasta_path, ref_contig, alt_contig = self.variant.synth_fasta(
            reference_fasta=self.cfg.reference, alleles=(1,1), flank=self.cfg.pileup.realigner_flank, dir=self.tempdir.name,
        )

        replicate_bam_path = simulate_variant_sequencing(
            fasta_path,
            2, # Allele count
            self.sample,
            reference=self.cfg.reference,
            shared_reference=self.cfg.shared_reference,
            dir=self.tempdir.name, #".",
            stats_path=os.path.join(FILE_DIR, "stats.json")
        )
        self.assertTrue(os.path.exists(replicate_bam_path))

    def test_gnomad_normalized_coverage(self):
        fasta_path, ref_contig, alt_contig = self.variant.synth_fasta(
            reference_fasta=self.cfg.reference, alleles=(1,1), flank=self.cfg.pileup.realigner_flank, dir=self.tempdir.name,
        )
        covg_path, *_ = self.variant.gnomad_coverage_profile(
            os.path.join(FILE_DIR, "1_896922_903086.gnomad.genomes.coverage.summary.tsv.gz"),
            ref_contig=ref_contig,
            alt_contig=alt_contig,
            flank=self.cfg.pileup.realigner_flank,
            dir=self.tempdir.name,
        )
        self.assertTrue(os.path.exists(covg_path))

        replicate_bam_path = simulate_variant_sequencing(
            fasta_path,
            2, # Allele count
            self.sample,
            reference=self.cfg.reference,
            shared_reference=self.cfg.shared_reference,
            dir=".", #self.tempdir.name, #".",
            gnomad_covg_path=covg_path,
        )
        self.assertTrue(os.path.exists(replicate_bam_path))

