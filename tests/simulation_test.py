import argparse, io, os, sys, tempfile, unittest
from unittest.mock import patch
from npsv2.simulation import RandomVariants

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")

class GenerateRandomDeletions(unittest.TestCase):
    def setUp(self):
        self.generator = RandomVariants(
            os.path.join(FILE_DIR, "1_896922_902998.fasta"), 
            os.path.join(FILE_DIR, "test_exclude.bed.gz"),
        )
 
    def test_deletion_generator(self):
        variants = list(self.generator._generate_deletions(100, n=2))
        self.assertEqual(len(variants), 2)

        for variant in variants:
            self.assertEqual(variant.length_change(), -100)
