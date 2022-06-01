import os, unittest
import pysam
from npsv2.range import Range
from npsv2.pileup import AlleleAssignment, BaseAlignment, ReadPileup, PileupRead

class PileupTest(unittest.TestCase):
    def setUp(self):
        # Get header to facilitate creating test reads
        self.header = pysam.AlignmentHeader.from_dict(
            {
                "HD": {"VN": "1.5", "SO": "coordinate"},
                "SQ": [{"SN": "1", "LN": 249250621}, {"SN": "2", "LN": 243199373}],
            }
        )

    def test_pileup_slices(self):
        pileup = ReadPileup(Range("1", 899721, 900192))
        read = pysam.AlignedSegment.fromstring(
            "HISEQ1:18:H8VC6ADXX:2:2213:9314:11990	161	1	899870	60	139M9S	2	33141319	0	GCTGGAGCCTGGGAAAGCGTGGCGCCCATGAATATCCGCAGGTCCGCAGTGGGGCTGCCGGGAGGGGTCCGCAGGTCCGCAGTGGGGCTGTGGGGGGGGGCCGCGCGTCCGCAGTGGGGGTGGGCTGCGGGAAGGGGGGGGCCGGGCC	@C@DFFFFGHHHHJIHGIJEGIJGIJIEEGJJIIEGGGDIIIHGIIIEEBCDFC?B@?=;BB&5;BD;51590;>8A:0505+:9ABBBD29A>.59@D.505&9&5&)090&5+:?<>&)5&58+589@D50<?<<B@D>&&)0<&)",
            header=self.header,
        )
        slices = pileup.read_columns(Range("1", 899721, 900192), PileupRead(read, AlleleAssignment.REF, 0, 0))
        
        self.assertEqual(list(slices), [(slice(148, 287), BaseAlignment.ALIGNED, slice(0,139)), (slice(287, 296),  BaseAlignment.SOFT_CLIP, slice(139,148))])


    def test_pileup_slices_larger_read(self):
        pileup = ReadPileup(Range("1", 899880, 900192))
        read = pysam.AlignedSegment.fromstring(
            "HISEQ1:18:H8VC6ADXX:2:2213:9314:11990	161	1	899870	60	139M9S	2	33141319	0	GCTGGAGCCTGGGAAAGCGTGGCGCCCATGAATATCCGCAGGTCCGCAGTGGGGCTGCCGGGAGGGGTCCGCAGGTCCGCAGTGGGGCTGTGGGGGGGGGCCGCGCGTCCGCAGTGGGGGTGGGCTGCGGGAAGGGGGGGGCCGGGCC	@C@DFFFFGHHHHJIHGIJEGIJGIJIEEGJJIIEGGGDIIIHGIIIEEBCDFC?B@?=;BB&5;BD;51590;>8A:0505+:9ABBBD29A>.59@D.505&9&5&)090&5+:?<>&)5&58+589@D50<?<<B@D>&&)0<&)",
            header=self.header,
        )
        slices = pileup.read_columns(Range("1", 899880, 900192), PileupRead(read, AlleleAssignment.REF, 0, 0))
        
        self.assertEqual(list(slices), [(slice(0, 128), BaseAlignment.ALIGNED, slice(11,139)), (slice(128, 137), BaseAlignment.SOFT_CLIP, slice(139,148))])


    def test_pileup_slices_leading_softclip(self):
        pileup = ReadPileup(Range("1", 899721, 900192))
        read = pysam.AlignedSegment.fromstring(
            "HISEQ1:18:H8VC6ADXX:1:2215:18684:93585	83	1	900126	0	76S72M	=	899656	-542	GCGGGGGGTCGGCGGGGGGGCTGGGGGAGGGGTGTGGGCGTCGGCATTGGGGAGGTGTTGTGGCAAGGGGGGGGCTGGGTCGGCAGGGGGGATGTGCTGGCGGGGGGGGGGGGGGGGTCCGCAGTGGGGATGTGCTGCCGGGAGGGGG	)&&&())&&&&)&&0&&+2(+&<0080(3<<2((&)&55))&(((((39++(((2(+(2(+9((100)-0)(((+(5-0)&(+(57)289>((83(&70&)55)&0)0',,''((67)))0*3FBD9C3C@;<1@1A@@21DD?A@@=",
            header=self.header,
        )
        slices = pileup.read_columns(Range("1", 899721, 900192), PileupRead(read, AlleleAssignment.REF, 0, 0))
        self.assertEqual(list(slices), [(slice(328, 404), BaseAlignment.SOFT_CLIP, slice(0,76)), (slice(404, 471), BaseAlignment.ALIGNED, slice(76,143))])


    @unittest.skipUnless(os.path.exists("/data/human_g1k_v37.fasta"), "Reference genome not available")
    def test_mismatch(self):
        pileup = ReadPileup(Range("1", 899721, 900192))
        
        with pysam.FastaFile("/data/human_g1k_v37.fasta") as ref_fasta:
            ref_seq = ref_fasta.fetch(reference="1", start=899721, end=900192)
       
        read = pysam.AlignedSegment.fromstring(
            "HISEQ1:18:H8VC6ADXX:2:2213:9314:11990	161	1	899870	60	139M9S	2	33141319	0	GCTGGAGCCTGGGAAAGCGTGGCGCCCATGAATATCCGCAGGTCCGCAGTGGGGCTGCCGGGAGGGGTCCGCAGGTCCGCAGTGGGGCTGTGGGGGGGGGCCGCGCGTCCGCAGTGGGGGTGGGCTGCGGGAAGGGGGGGGCCGGGCC	@C@DFFFFGHHHHJIHGIJEGIJGIJIEEGJJIIEGGGDIIIHGIIIEEBCDFC?B@?=;BB&5;BD;51590;>8A:0505+:9ABBBD29A>.59@D.505&9&5&)090&5+:?<>&)5&58+589@D50<?<<B@D>&&)0<&)",
            header=self.header,
        )
        slices = pileup.read_columns(Range("1", 899721, 900192), PileupRead(read, AlleleAssignment.REF, 0, 0), ref_seq)
        col_slice, align, read_slice = next(slices)
        
        self.assertEqual(col_slice, slice(148, 287))
        self.assertEqual(read_slice, slice(0,139))

        for read_idx, pileup_idx in enumerate(range(148, 287)): # Matched bases
            base = align[read_idx]
            if read.query_sequence[read_idx] == ref_seq[pileup_idx]:
                self.assertEqual(base, BaseAlignment.MATCH)
            else:
                self.assertEqual(base, BaseAlignment.MISMATCH)
