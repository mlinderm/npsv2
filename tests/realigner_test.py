import argparse, os, tempfile, unittest
from collections import Counter
from unittest.mock import patch
import pysam
from npsv2.variant import Variant
from npsv2.realigner import (
    FragmentRealigner,
    realign_fragment,
    AlleleAssignment,
    test_score_alignment,
    test_realign_read_pair,
)
from npsv2.pileup import FragmentTracker

FILE_DIR = os.path.join(os.path.dirname(__file__), "data")


class RealignerTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.params = argparse.Namespace(tempdir=self.tempdir.name, fragment_mean=569, fragment_sd=163)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_alignment_scoring(self):
        try:
            # Create SAM file with a single read
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sam") as sam_file:
                # fmt: off
                print("@HD", "VN:1.3", "SO:coordinate", sep="\t", file=sam_file)
                print("@SQ", "SN:1", "LN:249250621", sep="\t", file=sam_file)
                print("@RG", "ID:synth1", "LB:synth1", "PL:illumina", "PU:ART", "SM:HG002", sep="\t", file=sam_file)
                print(
                    "ref-354", "147", "1", "1", "60", "148M", "=", "2073433", "-435",
                    "AGCAGCCGAAGCGCCTCCTTTCACTCTAGGGTCCAGGCATCCAGCAGCCGAAGCGCCTCCTTTCAATCCAGGGTCCACACATCCAGCAGCCGAAGCGCCCTCCTTTCAATCCAGGGTCCAGGCATCTAGCAGCCGAAGCGCCTCCTTT",
                    "GG8CCGGGCGGGCGGGGGCGGCGGGGGGGGGGGCGGCGGGG=GGGJCCJGGGGGGCGGGGGGCG1GGCGG8GGCGC1GGCGJGCCGGJGJGJGGCGCJGJGJJCGGJJCJJGJJGJJJGJGCJJJGGJJJJGJJJGGGCGGGCGGCCC",
                    "RG:Z:synth1",
                    sep="\t",
                    file=sam_file,
                )
                # fmt: on

            # Read was aligned to very beginning of reference, so using read as reference should be all matches
            ref_sequence = "AGCAGCCGAAGCGCCTCCTTTCACTCTAGGGTCCAGGCATCCAGCAGCCGAAGCGCCTCCTTTCAATCCAGGGTCCACACATCCAGCAGCCGAAGCGCCCTCCTTTCAATCCAGGGTCCAGGCATCTAGCAGCCGAAGCGCCTCCTTT"
            scores = test_score_alignment(ref_sequence, sam_file.name)
            self.assertEqual(len(scores), 1)
            self.assertTrue(-9 < scores[0] < -8)
        finally:
            os.remove(sam_file.name)

    def test_alignment_scoring_with_iupac_ref_seq(self):
        try:
            # Create SAM file with a single read
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sam") as sam_file:
                # fmt: off
                print("@HD", "VN:1.3", "SO:coordinate", sep="\t", file=sam_file)
                print("@SQ", "SN:1", "LN:249250621", sep="\t", file=sam_file)
                print("@RG", "ID:synth1", "LB:synth1", "PL:illumina", "PU:ART", "SM:HG002", sep="\t", file=sam_file)
                print(
                    "ref-354", "147", "1", "1", "60", "148M", "=", "2073433", "-435",
                    "AGCAGCCGAAGCGCCTCCTTTCACTCTAGGGTCCAGGCATCCAGCAGCCGAAGCGCCTCCTTTCAATCCAGGGTCCACACATCCAGCAGCCGAAGCGCCCTCCTTTCAATCCAGGGTCCAGGCATCTAGCAGCCGAAGCGCCTCCTTT",
                    "GG8CCGGGCGGGCGGGGGCGGCGGGGGGGGGGGCGGCGGGG=GGGJCCJGGGGGGCGGGGGGCG1GGCGG8GGCGC1GGCGJGCCGGJGJGJGGCGCJGJGJJCGGJJCJJGJJGJJJGJGCJJJGGJJJJGJJJGGGCGGGCGGCCC",
                    "RG:Z:synth1",
                    sep="\t",
                    file=sam_file,
                )
                # fmt: on

            # Read is aligned to very beginning of reference, so using read as reference should be all matches (with IUPAC code)
            iupac_ref_sequence = "WGCAGCCGAAGCGCCTCCTTTCACTCTAGGGTCCAGGCATCCAGCAGCCGAAGCGCCTCCTTTCAATCCAGGGTCCACACATCCAGCAGCCGAAGCGCCCTCCTTTCAATCCAGGGTCCAGGCATCTAGCAGCCGAAGCGCCTCCTTT"
            
            scores = test_score_alignment(iupac_ref_sequence, sam_file.name)
            self.assertEqual(len(scores), 1)
            self.assertTrue(-9 < scores[0] < -8)  # Should be the same score as above
        finally:
            os.remove(sam_file.name)

    def test_realign_read_pair(self):
        # FASTA has a 3000bp flank
        fasta_path = os.path.join(FILE_DIR, "1_899922_899992_DEL.fasta")
        breakpoints = [("ref:3000-3001", "ref:3070-3071", "alt:3000-3001", "")]

        header = pysam.AlignmentHeader.from_dict(
            {"HD": {"VN": "1.5", "SO": "coordinate"}, "SQ": [{"SN": "1", "LN": 249250621}],}
        )

        read1 = pysam.AlignedSegment.fromstring(
            "HISEQ1:18:H8VC6ADXX:1:2103:1867:53768	163	1	899944	60	67M1I47M33S	=	900366	570	GTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGCTGTGCTGCGGGAAGGGGGGGGCCGGGCCCGCAGTGGGGATGTGCTGCCGGGAGGGGGGCGCGGGTCCGCGGGGGGGCGGGGCCGCCGGCGGGGGGGCGCGGG	CCCFFFFFHFHHGHIIIJIJGEHIIHIGIGIDGBEEBC/>;>C?:;@B<CA@C37@B6&8?BDBDD@505;@&50&0)9(+9?>(08(4:(4++055&005)05.&0)&5058&&)&&&)93&&&&&)&&)0&)&&)&&&&&&&&)&)",
            header=header,
        )
        read2 = pysam.AlignedSegment.fromstring(
            "HISEQ1:18:H8VC6ADXX:1:2103:1867:53768	83	1	900366	60	148M	=	899944	-570	TGGACGGATGGTTGTACGCCGTGGGGGGTAACGACGGTAGCTCCAGCCTCAACTCCATCGAGAAGTACAACCCGAGGACCAACAAGTGGGTGGCCGCATCCTGCATGTTCACCCGGCGCAGCAGTGTGGGTGTGGCGGTGCTGGAGCT	8>825AA:@DC@?950555B99@DDDDDDDDDB@BCDDBCBBDDDBCDDDCDDCDEDCDDDDEEEDDDDDDDBDCCC?DDDDDDDDDDDDBDDBBDDEDEFFFHHHHFHHBJJJJJIJJJJJJJJJJJJJJJJJJHHHHHFFFFFC@C",
            header=header,
        )

        read1_seq = read1.query_sequence
        read1_qual = "".join([chr(c) for c in read1.query_qualities])
        self.assertEqual(len(read1_seq), len(read1_qual))

        read2_seq = read2.query_sequence
        read2_qual = "".join([chr(c) for c in read2.query_qualities])
        self.assertEqual(len(read2_seq), len(read2_qual))

        results = test_realign_read_pair(
            fasta_path,
            breakpoints,
            read1.query_name,
            read1_seq,
            read1_qual,
            read2_seq=read2_seq,
            read2_qual=read2_qual,
            fragment_mean=self.params.fragment_mean,
            fragment_sd=self.params.fragment_sd,
            offset=0,  # Conversion already performed by pySAM
        )
        #print(results)

    def test_realign_reads(self):
        fasta_path = os.path.join(FILE_DIR, "1_899922_899992_DEL.fasta")
        breakpoints = [("ref:3000-3001", "ref:3070-3071", "alt:3000-3001", "")]
        bam_path = os.path.join(FILE_DIR, "1_896922_902998.bam")

        fragments = FragmentTracker()
        with pysam.AlignmentFile(bam_path, "rb") as bam_file:
            for read in bam_file:
                if (
                    read.is_duplicate
                    or read.is_qcfail
                    or read.is_unmapped
                    or read.is_secondary
                    or read.is_supplementary
                ):
                    # TODO: Potentially recover secondary/supplementary alignments if primary is outside pileup region
                    continue
                fragments.add_read(read)

        realigner = FragmentRealigner(fasta_path, breakpoints, self.params.fragment_mean, self.params.fragment_sd)

        allele_counts = Counter()
        for fragment in fragments:
            realignment = realign_fragment(realigner, fragment, assign_delta=1.0)
            allele_counts[realignment.allele] += 1

        self.assertEqual(allele_counts[AlleleAssignment.REF], 12)
        self.assertEqual(allele_counts[AlleleAssignment.ALT], 8)
