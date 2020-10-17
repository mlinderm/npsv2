import argparse, io, os, tempfile, unittest
from unittest.mock import patch
import pysam
from npsv2.variant import Variant
from npsv2.range import Range

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")

class SequenceResolvedDELVariantTestSuite(unittest.TestCase):
    def setUp(self):
        record = next(pysam.VariantFile(os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")))
        self.variant = Variant.from_pysam(record)

    def test_properties(self):
        # Range is 0-indexed half-open...
        self.assertEqual(self.variant.reference_range, Range("1",899921,899992))
        self.assertTrue(self.variant.is_biallelic())
