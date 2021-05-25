import argparse, io, os, sys, tempfile, unittest
from unittest.mock import patch
import pysam
from npsv2.variant import Variant, _reference_sequence
from npsv2.range import Range

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")


class SequenceResolvedDELVariantTestSuite(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        
        record = next(pysam.VariantFile(os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")))
        self.variant = Variant.from_pysam(record)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_properties(self):
        self.assertTrue(self.variant._sequence_resolved)
        self.assertEqual(self.variant._padding, 1)
        self.assertEqual(self.variant.ref_length, 71)
        self.assertEqual(self.variant.alt_length, 1)

        # Range is 0-indexed half-open...
        self.assertTrue(self.variant.is_biallelic())

    def test_region_strings(self):
        self.assertEqual(self.variant.reference_region, Range("1", 899922, 899992))
        self.assertEqual(self.variant.left_flank_region(left_flank=2, right_flank=5), Range("1", 899920, 899927))
        self.assertEqual(self.variant.right_flank_region(left_flank=2, right_flank=5), Range("1", 899990, 899997))

    @patch("npsv2.variant._reference_sequence", return_value="GGCTGCGGGGAGGGGGGCGCGGGTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGATGTG")
    def test_consensus_fasta(self, mock_ref):
            fasta_path, ref_contig, alt_contig = self.variant.synth_fasta(reference_fasta=None, dir=self.tempdir.name, line_width=sys.maxsize)
            self.assertEqual(ref_contig, "1_899922_899993")
            self.assertEqual(alt_contig, "1_899922_899993_alt")
            mock_ref.assert_called_once_with(None, Range("1",899921,899993), snv_vcf_path=None)

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

    def test_window_regions(self):
        regions = self.variant.window_regions(50, 1)
        self.assertEqual(len(regions), 5)
        for region in regions:
            self.assertEqual(region.length, 50)
        expected = ['1:899848-899897', '1:899898-899947', '1:899933-899982', '1:899968-900017', '1:900018-900067']
        self.assertEqual(regions, [Range.parse_literal(r) for r in expected])

    def test_window_regions_without_center(self):
        regions = self.variant.window_regions(100, 1)
        self.assertEqual(len(regions), 4)
        expected = ['1:899773-899872', '1:899873-899972', '1:899943-900042', '1:900043-900142']
        self.assertEqual(regions, [Range.parse_literal(r) for r in expected])

    def test_window_regions_breakpoints_only(self):
        regions = self.variant.window_regions(10, 0, window_interior=False)
        self.assertEqual(len(regions), 2)
        expected = ['1:899918-899927', '1:899988-899997']
        self.assertEqual(regions, [Range.parse_literal(r) for r in expected])

    def test_breakpoints(self):
        self.assertEqual(self.variant.ref_breakpoints(flank=1, contig="ref"), (Range.parse_literal("ref:1-2"), Range.parse_literal("ref:71-72")))
        self.assertEqual(self.variant.alt_breakpoints(flank=1, contig="alt"), (Range.parse_literal("alt:1-2"), None))

    def test_gnomad_coverage_profile(self):
        covg_path, ref_contig, alt_contig = self.variant.gnomad_coverage_profile(
            os.path.join(FILE_DIR, "1_896922_903086.gnomad.genomes.coverage.summary.tsv.gz"),
            flank=1,
            line_width=sys.maxsize,
            dir=self.tempdir.name,
        )
        self.assertEqual(ref_contig, "1_899922_899993")
        self.assertEqual(alt_contig, "1_899922_899993_alt")
        with open(covg_path, "r") as fasta:
            lines = [line.strip() for line in fasta]
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("1_899922_899993\t=") and lines[0].endswith("1"))
        self.assertEqual(lines[1], "1_899922_899993_alt\t=1")


class SymbolicDELVariantTestSuite(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

        vcf_path = os.path.join(self.tempdir.name, "test.vcf")
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
        self.assertEqual(self.variant.ref_length, 71)
        self.assertEqual(self.variant.alt_length, 1)


    @patch("npsv2.variant._reference_sequence", return_value="GGCTGCGGGGAGGGGGGCGCGGGTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGATGTG")
    def test_consensus_fasta(self, mock_ref):
            fasta_path, ref_contig, alt_contig = self.variant.synth_fasta(reference_fasta=None, dir=self.tempdir.name, line_width=sys.maxsize)
            self.assertEqual(ref_contig, "1_899922_899993")
            self.assertEqual(alt_contig, "1_899922_899993_alt")
            mock_ref.assert_called_once_with(None, Range("1",899921,899993), snv_vcf_path=None)

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

    def test_breakpoints(self):
        self.assertEqual(self.variant.ref_breakpoints(flank=1, contig="ref"), (Range.parse_literal("ref:1-2"), Range.parse_literal("ref:71-72")))
        self.assertEqual(self.variant.alt_breakpoints(flank=1, contig="alt"), (Range.parse_literal("alt:1-2"), None))


class ComplexDELVariantTestSuite(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        vcf_path = os.path.join(self.tempdir.name, "test.vcf")
        with open(vcf_path, "w") as vcf_file:
            vcf_file.write(
                """##fileformat=VCFv4.1
##INFO=<ID=CIEND,Number=2,Type=Integer,Description="Confidence interval around END for imprecise variants">
##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="Confidence interval around POS for imprecise variants">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant described in this record">
##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Difference in length between REF and ALT alleles">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##ALT=<ID=DEL,Description="Deletion">
##contig=<ID=4,length=191154276>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
4	20473846	HG3_PB_assemblyticsfalcon_19066	TATATATATATAGATCTATATATCTATATATAGATCTATATATAGATATATATCTATATATATAGATATATAGATATATAGATCTATATATAGATATATATATCTATATATAGATCTATATATAGATATAGATATCTATATAGATATCTATATCTATATATATGTAGATATATAGATATAGATATCTATATATCTATATATATAGATATCTATAGATATATATCTATATAGATATATCTATATCTATATATAGATATATATCTATATATAGATATATATCTATATATAGATAGATATATATCTATATATAGATATATCTATATCTATATATAGATATATATCTATATATAGATATATCTATATATAGATATATATCTATAGATATATCTATATATATCGATATATCTATATATATCGATATATA	ATATATATAGATATATCTATATATATCTATATAGATATATCTATATCTATATAGATATATCTATATATATATAGATATATCTATATCTATATAGATATATATCTATATATATATCTATATAGATATATCTATATAGATATAGATATATATCTATATATAGATATAGATATATCTATATAGATATATATCTATAGATATCTATATATATAGATATATAGATATCTATATCTATAT	10	PASS	END=20474269;SVTYPE=DEL;SVLEN=-190
"""
            )
        record = next(pysam.VariantFile(vcf_path))
        self.variant = Variant.from_pysam(record)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_variant_properties(self):
        self.assertEqual(
            self.variant.ref_length, len("TATATATATATAGATCTATATATCTATATATAGATCTATATATAGATATATATCTATATATATAGATATATAGATATATAGATCTATATATAGATATATATATCTATATATAGATCTATATATAGATATAGATATCTATATAGATATCTATATCTATATATATGTAGATATATAGATATAGATATCTATATATCTATATATATAGATATCTATAGATATATATCTATATAGATATATCTATATCTATATATAGATATATATCTATATATAGATATATATCTATATATAGATAGATATATATCTATATATAGATATATCTATATCTATATATAGATATATATCTATATATAGATATATCTATATATAGATATATATCTATAGATATATCTATATATATCGATATATCTATATATATCGATATATA")
        )

        self.assertEqual(
            self.variant.alt_length,
            len(
                "ATATATATAGATATATCTATATATATCTATATAGATATATCTATATCTATATAGATATATCTATATATATATAGATATATCTATATCTATATAGATATATATCTATATATATATCTATATAGATATATCTATATAGATATAGATATATATCTATATATAGATATAGATATATCTATATAGATATATATCTATAGATATCTATATATATAGATATATAGATATCTATATCTATAT"
            ),
        )

    def test_breakpoints(self):
        self.assertEqual(self.variant.left_flank_region(left_flank=1, right_flank=1), Range.parse_literal("4:20473845-20473846"))
        self.assertEqual(self.variant.right_flank_region(left_flank=1, right_flank=1), Range.parse_literal("4:20474269-20474270"))
        self.assertEqual(self.variant.ref_breakpoints(flank=1), (Range.parse_literal("4:1-2"), Range.parse_literal("4:425-426")))
        self.assertEqual(self.variant.alt_breakpoints(flank=1), (Range.parse_literal("4:1-2"), Range.parse_literal("4:235-236")))

    @patch("npsv2.variant._reference_sequence", return_value="GTATATATATATAGATCTATATATCTATATATAGATCTATATATAGATATATATCTATATATATAGATATATAGATATATAGATCTATATATAGATATATATATCTATATATAGATCTATATATAGATATAGATATCTATATAGATATCTATATCTATATATATGTAGATATATAGATATAGATATCTATATATCTATATATATAGATATCTATAGATATATATCTATATAGATATATCTATATCTATATATAGATATATATCTATATATAGATATATATCTATATATAGATAGATATATATCTATATATAGATATATCTATATCTATATATAGATATATATCTATATATAGATATATCTATATATAGATATATATCTATAGATATATCTATATATATCGATATATCTATATATATCGATATATAT")
    def test_consensus_fasta(self, mock_ref):
        fasta_path, ref_contig, alt_contig = self.variant.synth_fasta(reference_fasta=None, dir=self.tempdir.name, line_width=sys.maxsize)
        mock_ref.assert_called_once_with(None, Range.parse_literal("4:20473845-20474270"), snv_vcf_path=None)

        with open(fasta_path, "r") as fasta:
            lines = [line.strip() for line in fasta]
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], ">4_20473845_20474270")
        self.assertEqual(lines[0], f">{ref_contig}")
        self.assertEqual(
            lines[1],
            "GTATATATATATAGATCTATATATCTATATATAGATCTATATATAGATATATATCTATATATATAGATATATAGATATATAGATCTATATATAGATATATATATCTATATATAGATCTATATATAGATATAGATATCTATATAGATATCTATATCTATATATATGTAGATATATAGATATAGATATCTATATATCTATATATATAGATATCTATAGATATATATCTATATAGATATATCTATATCTATATATAGATATATATCTATATATAGATATATATCTATATATAGATAGATATATATCTATATATAGATATATCTATATCTATATATAGATATATATCTATATATAGATATATCTATATATAGATATATATCTATAGATATATCTATATATATCGATATATCTATATATATCGATATATAT",
        )
        self.assertEqual(lines[2], ">4_20473845_20474270_alt")
        self.assertEqual(lines[2], f">{alt_contig}")
        self.assertEqual(
            lines[3],
            "GATATATATAGATATATCTATATATATCTATATAGATATATCTATATCTATATAGATATATCTATATATATATAGATATATCTATATCTATATAGATATATATCTATATATATATCTATATAGATATATCTATATAGATATAGATATATATCTATATATAGATATAGATATATCTATATAGATATATATCTATAGATATCTATATATATAGATATATAGATATCTATATCTATATT",
        )

class SingleBaseComplexDELVariantTestSuite(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        vcf_path = os.path.join(self.tempdir.name, "test.vcf")
        with open(vcf_path, "w") as vcf_file:
            vcf_file.write(
                """##fileformat=VCFv4.1
##INFO=<ID=CIEND,Number=2,Type=Integer,Description="Confidence interval around END for imprecise variants">
##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="Confidence interval around POS for imprecise variants">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant described in this record">
##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Difference in length between REF and ALT alleles">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##ALT=<ID=DEL,Description="Deletion">
##contig=<ID=8,length=146364022>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
8	79683398	.	AACCTCCCAACGCAATAGACATTGTGGTTTTCATTGCATATCATTCCTATTTCTCTCTCTCCATTATTTAGCAGTAATTTTTTTAATGAA	C	20	PASS	END=79683487;SVTYPE=DEL;SVLEN=-89
"""
            )
        record = next(pysam.VariantFile(vcf_path))
        self.variant = Variant.from_pysam(record)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_breakpoints(self):
        self.assertEqual(self.variant.left_flank_region(left_flank=1, right_flank=1), Range.parse_literal("8:79683397-79683398"))
        self.assertEqual(self.variant.right_flank_region(left_flank=1, right_flank=1), Range.parse_literal("8:79683487-79683488"))

        self.assertEqual(self.variant.ref_breakpoints(1, contig="ref"), (Range.parse_literal("ref:1-2"), Range.parse_literal("ref:91-92")))
        self.assertEqual(self.variant.alt_breakpoints(1, contig="alt"), (Range.parse_literal("alt:1-2"), Range.parse_literal("alt:2-3")))

    @patch("npsv2.variant._reference_sequence", return_value="AAACCTCCCAACGCAATAGACATTGTGGTTTTCATTGCATATCATTCCTATTTCTCTCTCTCCATTATTTAGCAGTAATTTTTTTAATGAAA")
    def test_consensus_fasta(self, mock_ref):
        fasta_path, ref_contig, alt_contig = self.variant.synth_fasta(reference_fasta=None, dir=self.tempdir.name, line_width=sys.maxsize)
        mock_ref.assert_called_once_with(None, Range.parse_literal("8:79683397-79683488"), snv_vcf_path=None)

        with open(fasta_path, "r") as fasta:
            lines = [line.strip() for line in fasta]
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], ">8_79683397_79683488")
        self.assertEqual(lines[0], f">{ref_contig}")
        self.assertEqual(
            lines[1],
            "AAACCTCCCAACGCAATAGACATTGTGGTTTTCATTGCATATCATTCCTATTTCTCTCTCTCCATTATTTAGCAGTAATTTTTTTAATGAAA",
        )
        self.assertEqual(lines[2], ">8_79683397_79683488_alt")
        self.assertEqual(lines[2], f">{alt_contig}")
        self.assertEqual(
            lines[3],
            "ACA",
        )

class IncompleteSequenceResolvedDELVariantTestSuite(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        vcf_path = os.path.join(self.tempdir.name, "test.vcf")
        with open(vcf_path, "w") as vcf_file:
            vcf_file.write(
                """##fileformat=VCFv4.1
##INFO=<ID=CIEND,Number=2,Type=Integer,Description="Confidence interval around END for imprecise variants">
##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="Confidence interval around POS for imprecise variants">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant described in this record">
##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Difference in length between REF and ALT alleles">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##ALT=<ID=DEL,Description="Deletion">
##FILTER=<ID=LongReadHomRef,Description="Long reads supported homozygous reference for all individuals">
##contig=<ID=1,length=249250621>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
1	67808460	HG2_10X_SVrefine210Xhap12_405	TGAGACAGGGTGTCATTCTGTCCCCCAGGCTGAAGTGTGGTGGCACAATCTCAGCTCACTGCAGCCTTCACCTCCTATGCTCAAGTGATCCTCCCACCTCAGCCTCCCAAGTAGCTGAGACTACAGGCATCCATCACCACGCCCAGCTAATTTTTGTTTGTCACA	T	20	LongReadHomRef	SVTYPE=DEL;SVLEN=-164
"""
            )
        record = next(pysam.VariantFile(vcf_path))
        self.variant = Variant.from_pysam(record)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_properties(self):
        self.assertTrue(self.variant._sequence_resolved)
        self.assertEqual(self.variant._padding, 1)
        self.assertEqual(self.variant.ref_length, 165)
        self.assertEqual(self.variant.alt_length, 1)
        self.assertEqual(self.variant.end, 67808624) # 0-indexed, half-open

@unittest.skipUnless(os.path.exists("/data/human_g1k_v37.fasta"), "Reference genome not available")
class FASTAWithIUPACCodesDELVariant(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        record = next(pysam.VariantFile(os.path.join(FILE_DIR, "1_67808460_67808624_DEL.vcf.gz")))
        self.variant = Variant.from_pysam(record)
        self.snv_vcf_path = os.path.join(FILE_DIR, "1_67806460_67811624.snvs.vcf.gz")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_reference_sequence_with_IUPAC(self):
        region = self.variant.reference_region.expand(1000)
        ref_seq = _reference_sequence("/data/human_g1k_v37.fasta", region, self.snv_vcf_path)
        self.assertEqual(len(ref_seq), region.length)
        self.assertEqual(ref_seq[67808441 - 1 - region.start], "Y")  # Two SNVs (recall POS is 1-indexed) in region
        self.assertEqual(ref_seq[67808536 - 1 - region.start], "R")
    
    def test_consensus_fasta(self):
        region = self.variant.reference_region.expand(30)
        fasta_path, ref_contig, alt_contig = self.variant.synth_fasta(
            reference_fasta="/data/human_g1k_v37.fasta", 
            dir=self.tempdir.name,
            line_width=sys.maxsize,
            flank=30,
            snv_vcf_path=self.snv_vcf_path,
        )
        with open(fasta_path, "r") as fasta:
            lines = [line.strip() for line in fasta]
        self.assertEqual(len(lines), 4)
        ref_seq = lines[1]
        self.assertEqual(ref_seq[67808441 - 1 - region.start], "Y")  # Two SNVs (recall POS is 1-indexed) in region
        self.assertEqual(ref_seq[67808536 - 1 - region.start], "R")
            