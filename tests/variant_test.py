import argparse, io, os, sys, tempfile, unittest
from unittest.mock import patch
import pysam
from npsv2.variant import Variant
from npsv2.range import Range

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")


class SequenceResolvedDELVariantTestSuite(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.params = argparse.Namespace(tempdir=self.tempdir.name)
        
        record = next(pysam.VariantFile(os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")))
        self.variant = Variant.from_pysam(record)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_properties(self):
        self.assertTrue(self.variant._sequence_resolved)
        self.assertEqual(self.variant._padding, 1)

        # Range is 0-indexed half-open...
        self.assertEqual(self.variant.reference_range, Range("1", 899922, 899992))
        self.assertTrue(self.variant.is_biallelic())

    @patch("npsv2.variant._reference_sequence", return_value="GGCTGCGGGGAGGGGGGCGCGGGTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGATGTG")
    def test_consensus_fasta(self, mock_ref):
            fasta_path, ref_contig, alt_contig = self.variant.synth_fasta(reference_fasta=None, dir=self.params.tempdir, line_width=sys.maxsize)
            self.assertEqual(ref_contig, "1_899922_899993")
            self.assertEqual(alt_contig, "1_899922_899993_alt")
            mock_ref.assert_called_once_with(None, Range("1",899921,899993))

            with open(fasta_path, "r") as fasta:
                lines = [line.strip() for line in fasta]
            self.assertEqual(len(lines), 4)
            self.assertEqual(lines[0], ">1_899922_899993")
            self.assertEqual(
                lines[1],
                "GGCTGCGGGGAGGGGGGCGCGGGTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGATGTG",
            )
            self.assertEqual(lines[2], ">1_899922_899993_alt")
            self.assertEqual(lines[3], "GG")

    def test_construct_proto(self):
        proto = self.variant.as_proto()
        self.assertEqual(proto.start, 899921)

class SymbolicDELVariantTestSuite(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.params = argparse.Namespace(tempdir=self.tempdir.name)

        vcf_path = os.path.join(self.params.tempdir, "test.vcf")
        with open(vcf_path, "w") as vcf_file:
            vcf_file.write(
                """##fileformat=VCFv4.1
##INFO=<ID=CIEND,Number=2,Type=Integer,Description="Confidence interval around END for imprecise variants">
##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="Confidence interval around POS for imprecise variants">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant described in this record">
##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Difference in length between REF and ALT alleles">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##ALT=<ID=DEL,Description="Deletion">
##contig=<ID=1,length=249250621>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
1	899922	.	G	<DEL>	.	PASS	END=899992;SVTYPE=DEL;SVLEN=-70
"""
            )
            
        record = next(pysam.VariantFile(vcf_path))
        self.variant = Variant.from_pysam(record)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_properties(self):
        self.assertFalse(self.variant._sequence_resolved)
        self.assertEqual(self.variant._padding, 1)

    @patch("npsv2.variant._reference_sequence", return_value="GGCTGCGGGGAGGGGGGCGCGGGTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGATGTG")
    def test_consensus_fasta(self, mock_ref):
            fasta_path, ref_contig, alt_contig = self.variant.synth_fasta(reference_fasta=None, dir=self.params.tempdir, line_width=sys.maxsize)
            self.assertEqual(ref_contig, "1_899922_899993")
            self.assertEqual(alt_contig, "1_899922_899993_alt")
            mock_ref.assert_called_once_with(None, Range("1",899921,899993))

            with open(fasta_path, "r") as fasta:
                lines = [line.strip() for line in fasta]
            self.assertEqual(len(lines), 4)
            self.assertEqual(lines[0], ">1_899922_899993")
            self.assertEqual(
                lines[1],
                "GGCTGCGGGGAGGGGGGCGCGGGTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGATGTG",
            )
            self.assertEqual(lines[2], ">1_899922_899993_alt")
            self.assertEqual(lines[3], "GG")
