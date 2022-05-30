import argparse, io, os, sys, tempfile, unittest
from unittest.mock import patch
import pysam
from npsv2.variant import Variant, _reference_sequence
from npsv2.range import Range
from npsv2 import npsv2_pb2

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")


class SequenceResolvedDELVariantTestSuite(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        
        record = next(pysam.VariantFile(os.path.join(FILE_DIR, "1_899922_899992_DEL.vcf.gz")))
        self.variant = Variant.from_pysam(record)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_properties(self):
        self.assertTrue(self.variant.is_deletion)
        self.assertTrue(self.variant._sequence_resolved)
        self.assertEqual(self.variant._padding, 1)
        self.assertEqual(self.variant.ref_length, 71)
        self.assertEqual(self.variant.alt_length(), 1)
        self.assertEqual(self.variant.length_change(), -70)
        self.assertEqual(set(self.variant.alt_allele_indices), {1})
        self.assertEqual(self.variant.num_alt, 1)
        self.assertTrue(self.variant.is_biallelic())

    def test_regions(self):
        self.assertEqual(self.variant.reference_region, Range("1", 899922, 899992))
        self.assertEqual(self.variant.left_flank_region(left_flank=2, right_flank=5), Range("1", 899920, 899927))
        self.assertEqual(self.variant.right_flank_region(left_flank=2, right_flank=5), Range("1", 899990, 899997))

    @patch("npsv2.variant._reference_sequence", return_value="GGCTGCGGGGAGGGGGGCGCGGGTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGATGTG")
    def test_consensus_fasta(self, mock_ref):
            fasta_path, ref_contig, alt_contig = self.variant.synth_fasta(reference_fasta=None, dir=self.tempdir.name, line_width=sys.maxsize)
            self.assertEqual(ref_contig, "1_899922_899993")
            self.assertEqual(alt_contig, "1_899922_899993_1_alt")
            mock_ref.assert_called_once_with(None, Range("1",899921,899993), snv_vcf_path=None)

            with open(fasta_path, "r") as fasta:
                lines = [line.strip() for line in fasta]
            self.assertEqual(len(lines), 4)
            self.assertEqual(lines[0], ">1_899922_899993")
            self.assertEqual(
                lines[1],
                "GGCTGCGGGGAGGGGGGCGCGGGTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGATGTG",
            )
            self.assertEqual(lines[2], ">1_899922_899993_1_alt")
            self.assertEqual(lines[3], "GG")

    def test_construct_proto(self):
        proto = self.variant.as_proto()
        self.assertEqual(proto.start, 899921)
        self.assertEqual(proto.svtype, npsv2_pb2.StructuralVariant.Type.DEL)

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
        self.assertEqual(self.variant.alt_length(), 1)


    @patch("npsv2.variant._reference_sequence", return_value="GGCTGCGGGGAGGGGGGCGCGGGTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGATGTG")
    def test_consensus_fasta(self, mock_ref):
            fasta_path, ref_contig, alt_contig = self.variant.synth_fasta(reference_fasta=None, dir=self.tempdir.name, line_width=sys.maxsize)
            self.assertEqual(ref_contig, "1_899922_899993")
            self.assertEqual(alt_contig, "1_899922_899993_1_alt")
            mock_ref.assert_called_once_with(None, Range("1",899921,899993), snv_vcf_path=None)

            with open(fasta_path, "r") as fasta:
                lines = [line.strip() for line in fasta]
            self.assertEqual(len(lines), 4)
            self.assertEqual(lines[0], ">1_899922_899993")
            self.assertEqual(
                lines[1],
                "GGCTGCGGGGAGGGGGGCGCGGGTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGATGTG",
            )
            self.assertEqual(lines[2], ">1_899922_899993_1_alt")
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
            self.variant.alt_length(),
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
        self.assertEqual(lines[2], ">4_20473845_20474270_1_alt")
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
        self.assertEqual(lines[2], ">8_79683397_79683488_1_alt")
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
        self.assertEqual(self.variant.alt_length(), 1)
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


class MultiallelicSequenceResolvedDELVariantTestSuite(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

        vcf_path = os.path.join(self.tempdir.name, "test.vcf")
        with open(vcf_path, "w") as vcf_file:
            vcf_file.write(
                """##fileformat=VCFv4.1
##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference in length between REF and ALT alleles">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for the ref and alt alleles in the order listed">
##ALT=<ID=DEL,Description="Deletion">
##contig=<ID=18,length=78077248>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG002
18	77499776	.	AGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCG	A,AGGACGGGTGGGACTCTCATACCCACGGCCG	30	.	SVTYPE=DEL;SVLEN=-360,-330	GT:AD	1|2:0,1,1
"""
            )
            
        record = next(pysam.VariantFile(vcf_path))
        self.variant = Variant.from_pysam(record)
    
    def tearDown(self):
        self.tempdir.cleanup()

    def test_create_variant(self):
        self.assertTrue(self.variant.is_deletion)
        self.assertTrue(self.variant._sequence_resolved)
        self.assertEqual(self.variant._padding, 1)
        self.assertEqual(self.variant._right_padding, [0, 30])
        
        self.assertEqual(self.variant.num_alt, 2)
        self.assertFalse(self.variant.is_biallelic())
        self.assertEqual(set(self.variant.alt_allele_indices), {1, 2})

        self.assertEqual(self.variant.end, 77500136)
        
        self.assertEqual(self.variant.length_change(allele=None), (-360, -330))
        self.assertEqual(self.variant.length_change(allele=1), -360)
        self.assertEqual(self.variant.length_change(allele=2), -330)
        self.assertEqual(self.variant.alt_length(1),1)
        self.assertEqual(self.variant.alt_length(2),31)
        
        self.assertEqual(self.variant.genotype_allele_count("HG002"), 2)
        self.assertEqual(self.variant.genotype_allele_count("HG002", {1}), 1)
        self.assertEqual(self.variant.genotype_allele_count("HG002", {2}), 1)
        
        self.assertEqual(self.variant.ref_breakpoints(1, allele=1), (Range("18",0,2), Range("18",360,362)))
        self.assertEqual(self.variant.ref_breakpoints(1, allele=2), (Range("18",0,2), Range("18",330,332)))
        
        self.assertEqual(self.variant.alt_breakpoints(1, allele=1), (Range("18",0,2), None))
        self.assertEqual(self.variant.alt_breakpoints(1, allele=2), (Range("18",0,2), None))

        self.assertEqual(self.variant.left_flank_region(1, allele=1), Range("18",77499775,77499776))
        self.assertEqual(self.variant.left_flank_region(1, allele=2), Range("18",77499775,77499776))
        
        self.assertEqual(self.variant.right_flank_region(1, allele=1), Range("18",77500136,77500137))
        self.assertEqual(self.variant.right_flank_region(1, allele=2), Range("18",77500106,77500107))

    def test_protobuf_creation(self):
        proto = self.variant.as_proto()
        self.assertEqual(proto.svlen, [-360,-330])

    @patch("npsv2.variant._reference_sequence", return_value="AGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGGGACGGGTGGGACTCTCATACCCACGGCCGG")
    def test_consensus_fasta(self, mock_ref):
            fasta_path, ref_contig, alt_contig = self.variant.synth_fasta(alleles=(1,2), reference_fasta=None, dir=self.tempdir.name, line_width=sys.maxsize)
            mock_ref.assert_called_once_with(None, Range("18",77499775,77500137), snv_vcf_path=None)
            self.assertEqual(ref_contig, "18_77499776_77500137")
            self.assertEqual(alt_contig, ["18_77499776_77500137_1_alt","18_77499776_77500137_2_alt"]) 
            
            with open(fasta_path, "r") as fasta:
                lines = [line.strip() for line in fasta]
            self.assertEqual(len(lines), 5)
            self.assertEqual(lines[0], f">{ref_contig}")
            self.assertEqual(lines[1], f">{alt_contig[0]}")
            self.assertEqual(lines[2], "AG")
            self.assertEqual(lines[3], f">{alt_contig[1]}")
            self.assertEqual(lines[4], "AGGACGGGTGGGACTCTCATACCCACGGCCGG")

class SequenceResolvedINSVariantTestSuite(unittest.TestCase):
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
##ALT=<ID=INS,Description="Insertion">
##contig=<ID=1,length=249250621>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
1	931634	HG2_PB_SVrefine2PB10Xhap12_17	A	AGGGAGGGCAGAAAGGACCCCCACGTGAGGGGGCACCCCACATCTGGGGCCACAGGATGCAGGGTGGGGAGGGCAGAAAGGCCCCCCCGCGGGAAGGGGCACCCCACATCTGGGCCACAGGATGCAGGGTGGGGAGGGCAGAAAGGCCCCCCCGCGGGAAGGGGCACCCCACATCTGGGGCCACAGGATGCAGGGTG	.	PASS	SVTYPE=INS;END=931634;SVLEN=196
"""
        )
        record = next(pysam.VariantFile(vcf_path))
        self.variant = Variant.from_pysam(record)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_properties(self):
        self.assertTrue(self.variant.is_insertion)
        self.assertTrue(self.variant._sequence_resolved)
        self.assertEqual(self.variant._padding, 1)
        self.assertEqual(self.variant.ref_length, 1)
        self.assertEqual(self.variant.alt_length(), 197)
        self.assertEqual(self.variant.length_change(), 196)
        self.assertEqual(set(self.variant.alt_allele_indices), {1})
        self.assertEqual(self.variant.num_alt, 1)
        self.assertTrue(self.variant.is_biallelic())

    def test_regions(self):
        self.assertEqual(self.variant.left_flank_region(left_flank=2, right_flank=5), Range("1", 931632, 931639))
        self.assertEqual(self.variant.right_flank_region(left_flank=2, right_flank=5), Range("1", 931632, 931639))

    def test_breakpoints(self):
        self.assertEqual(self.variant.ref_breakpoints(flank=1, contig="ref"), (Range.parse_literal("ref:1-2"), None))
        self.assertEqual(self.variant.alt_breakpoints(flank=1, contig="alt"), (Range.parse_literal("alt:1-2"), Range.parse_literal("alt:197-198")))

    @patch("npsv2.variant._reference_sequence", return_value="AG")
    def test_consensus_fasta(self, mock_ref):
            fasta_path, ref_contig, alt_contig = self.variant.synth_fasta(reference_fasta=None, dir=self.tempdir.name, line_width=sys.maxsize)
            mock_ref.assert_called_once_with(None, Range("1",931633,931635), snv_vcf_path=None)
            self.assertEqual(ref_contig, "1_931634_931635")
            self.assertEqual(alt_contig, "1_931634_931635_1_alt")
            
            with open(fasta_path, "r") as fasta:
                lines = [line.strip() for line in fasta]
            self.assertEqual(len(lines), 4)
            self.assertEqual(lines[0], f">{ref_contig}")
            self.assertEqual(lines[1], "AG")
            self.assertEqual(lines[2], f">{alt_contig}")
            self.assertEqual(
                lines[3],
                "AGGGAGGGCAGAAAGGACCCCCACGTGAGGGGGCACCCCACATCTGGGGCCACAGGATGCAGGGTGGGGAGGGCAGAAAGGCCCCCCCGCGGGAAGGGGCACCCCACATCTGGGCCACAGGATGCAGGGTGGGGAGGGCAGAAAGGCCCCCCCGCGGGAAGGGGCACCCCACATCTGGGGCCACAGGATGCAGGGTGG",
            )

    def test_construct_proto(self):
        proto = self.variant.as_proto()
        self.assertEqual(proto.start, 931633)
        self.assertEqual(proto.svtype, npsv2_pb2.StructuralVariant.Type.INS)


class SNVVariantTestSuite(unittest.TestCase):
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
##ALT=<ID=INS,Description="Insertion">
##contig=<ID=1,length=249250621>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
1	900298	.	C	G	299.17	PASS	.
"""
        )
        record = next(pysam.VariantFile(vcf_path))
        self.variant = Variant.from_pysam(record)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_properties(self):
        self.assertTrue(self.variant.is_substitution)
        self.assertTrue(self.variant.is_SNV)
        self.assertFalse(self.variant.is_MNV)
        self.assertTrue(self.variant._sequence_resolved)
        self.assertEqual(self.variant._padding, 0)
        self.assertEqual(self.variant.ref_length, 1)
        self.assertEqual(self.variant.alt_length(), 1)
        self.assertEqual(self.variant.length_change(), 0)
        self.assertEqual(set(self.variant.alt_allele_indices), {1})
        self.assertEqual(self.variant.num_alt, 1)
        self.assertTrue(self.variant.is_biallelic())
        self.assertEqual(self.variant.name, "1_900298_900298_SUB")

    def test_regions(self):
        self.assertEqual(self.variant.left_flank_region(left_flank=2, right_flank=5), Range("1", 900295, 900302))
        self.assertEqual(self.variant.right_flank_region(left_flank=2, right_flank=5), Range("1", 900296, 900303))

    def test_breakpoints(self):
        self.assertEqual(self.variant.ref_breakpoints(flank=1, contig="ref"), (Range.parse_literal("ref:1-2"), Range.parse_literal("ref:2-3")))
        self.assertEqual(self.variant.alt_breakpoints(flank=1, contig="alt"), (Range.parse_literal("alt:1-2"), Range.parse_literal("alt:2-3")))

    def test_construct_proto(self):
        proto = self.variant.as_proto()
        self.assertEqual(proto.start, 900297)
        self.assertEqual(proto.end, 900298)
        self.assertEqual(proto.svtype, npsv2_pb2.StructuralVariant.Type.SUB)


class ReferenceSequenceTestSuite(unittest.TestCase):
    def setUp(self):
        self.vcf_path = os.path.join(FILE_DIR, "overlapping_snvs.vcf.gz")

    @patch("pysam.FastaFile")
    def test_overlapping_snvs(self, MockFastaFile):
        # FastaFile is used in a context
        MockFastaFile.return_value.__enter__.return_value.fetch.return_value = "C"
        # The alleles should be A,C,T or H
        self.assertEqual(_reference_sequence("null.fasta", Range.parse_literal("chr4:112580127-112580127"), self.vcf_path), "H")