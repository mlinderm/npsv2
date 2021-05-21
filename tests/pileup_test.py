import os, unittest
import pysam
from npsv2.range import Range
from npsv2.pileup import BaseAlignment, Pileup

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
        pileup = Pileup(Range("1", 899721, 900192))
        
        read = pysam.AlignedSegment.fromstring(
            "HISEQ1:18:H8VC6ADXX:2:2213:9314:11990	161	1	899870	60	139M9S	2	33141319	0	GCTGGAGCCTGGGAAAGCGTGGCGCCCATGAATATCCGCAGGTCCGCAGTGGGGCTGCCGGGAGGGGTCCGCAGGTCCGCAGTGGGGCTGTGGGGGGGGGCCGCGCGTCCGCAGTGGGGGTGGGCTGCGGGAAGGGGGGGGCCGGGCC	@C@DFFFFGHHHHJIHGIJEGIJGIJIEEGJJIIEGGGDIIIHGIIIEEBCDFC?B@?=;BB&5;BD;51590;>8A:0505+:9ABBBD29A>.59@D.505&9&5&)090&5+:?<>&)5&58+589@D50<?<<B@D>&&)0<&)",
            header=self.header,
        )
        read_start, slices = pileup.read_columns(read)
        
        self.assertEqual(read_start, 899869)
        self.assertEqual(slices, [(slice(148, 287), pysam.CMATCH, slice(0,139)), (slice(287, 296), pysam.CSOFT_CLIP, slice(139,148))])


    def test_pileup_slices_larger_read(self):
        pileup = Pileup(Range("1", 899880, 900192))
        
        read = pysam.AlignedSegment.fromstring(
            "HISEQ1:18:H8VC6ADXX:2:2213:9314:11990	161	1	899870	60	139M9S	2	33141319	0	GCTGGAGCCTGGGAAAGCGTGGCGCCCATGAATATCCGCAGGTCCGCAGTGGGGCTGCCGGGAGGGGTCCGCAGGTCCGCAGTGGGGCTGTGGGGGGGGGCCGCGCGTCCGCAGTGGGGGTGGGCTGCGGGAAGGGGGGGGCCGGGCC	@C@DFFFFGHHHHJIHGIJEGIJGIJIEEGJJIIEGGGDIIIHGIIIEEBCDFC?B@?=;BB&5;BD;51590;>8A:0505+:9ABBBD29A>.59@D.505&9&5&)090&5+:?<>&)5&58+589@D50<?<<B@D>&&)0<&)",
            header=self.header,
        )
        read_start, slices = pileup.read_columns(read)
        
        self.assertEqual(read_start, 899869)
        self.assertEqual(slices, [(slice(0, 128), pysam.CMATCH, slice(11,139)), (slice(128, 137), pysam.CSOFT_CLIP, slice(139,148))])


    def test_add_single_read(self):
        pileup = Pileup(Range("1", 899721, 900192))

        read = pysam.AlignedSegment.fromstring(
            "HISEQ1:18:H8VC6ADXX:2:2213:9314:11990	161	1	899870	60	139M9S	2	33141319	0	GCTGGAGCCTGGGAAAGCGTGGCGCCCATGAATATCCGCAGGTCCGCAGTGGGGCTGCCGGGAGGGGTCCGCAGGTCCGCAGTGGGGCTGTGGGGGGGGGCCGCGCGTCCGCAGTGGGGGTGGGCTGCGGGAAGGGGGGGGCCGGGCC	@C@DFFFFGHHHHJIHGIJEGIJGIJIEEGJJIIEGGGDIIIHGIIIEEBCDFC?B@?=;BB&5;BD;51590;>8A:0505+:9ABBBD29A>.59@D.505&9&5&)090&5+:?<>&)5&58+589@D50<?<<B@D>&&)0<&)",
            header=self.header,
        )
        pileup.add_read(read)

        for i in range(148): # Before the read
            self.assertEqual(len(pileup[i]), 0)
        for i in range(148, 287): # Matched bases
            column = pileup[i]
            self.assertTrue(column.aligned_bases, 1)
            self.assertEqual(len(column), 1)
        for i in range(287, 296): # Soft-clipped bases
            column = pileup[i]
            self.assertTrue(column.soft_clipped_bases, 1)
            self.assertEqual(len(column), 1)    
        for i in range(296, 471): # After the read
            self.assertEqual(len(pileup[i]), 0)
        

    def test_add_read_with_leading_soft_clip(self):
        read = pysam.AlignedSegment.fromstring(
            "HISEQ1:18:H8VC6ADXX:1:2215:18684:93585	83	1	900126	0	76S72M	=	899656	-542	GCGGGGGGTCGGCGGGGGGGCTGGGGGAGGGGTGTGGGCGTCGGCATTGGGGAGGTGTTGTGGCAAGGGGGGGGCTGGGTCGGCAGGGGGGATGTGCTGGCGGGGGGGGGGGGGGGGTCCGCAGTGGGGATGTGCTGCCGGGAGGGGG	)&&&())&&&&)&&0&&+2(+&<0080(3<<2((&)&55))&(((((39++(((2(+(2(+9((100)-0)(((+(5-0)&(+(57)289>((83(&70&)55)&0)0',,''((67)))0*3FBD9C3C@;<1@1A@@21DD?A@@=",
            header=self.header,
        )
        
        pileup = Pileup(Range("1", 899721, 900192))
        pileup.add_read(read)

        for i in range(328): # Before the read
            self.assertEqual(len(pileup[i]), 0)
        for i in range(328, 404): # Soft-clipped bases
            column = pileup[i]
            self.assertTrue(column.soft_clipped_bases, 1)
            self.assertEqual(len(column), 1)
        for i in range(404, 471): # Matched bases
            column = pileup[i]
            self.assertTrue(column.aligned_bases, 1)
            self.assertEqual(len(column), 1)

    @unittest.skipUnless(os.path.exists("/data/human_g1k_v37.fasta"), "Reference genome not available")
    def test_mismatch(self):
        pileup = Pileup(Range("1", 899721, 900192))
        
        with pysam.FastaFile("/data/human_g1k_v37.fasta") as ref_fasta:
            ref_seq = ref_fasta.fetch(reference="1", start=899721, end=900192)
        self.assertEqual(len(ref_seq), len(pileup))

        read = pysam.AlignedSegment.fromstring(
            "HISEQ1:18:H8VC6ADXX:2:2213:9314:11990	161	1	899870	60	139M9S	2	33141319	0	GCTGGAGCCTGGGAAAGCGTGGCGCCCATGAATATCCGCAGGTCCGCAGTGGGGCTGCCGGGAGGGGTCCGCAGGTCCGCAGTGGGGCTGTGGGGGGGGGCCGCGCGTCCGCAGTGGGGGTGGGCTGCGGGAAGGGGGGGGCCGGGCC	@C@DFFFFGHHHHJIHGIJEGIJGIJIEEGJJIIEGGGDIIIHGIIIEEBCDFC?B@?=;BB&5;BD;51590;>8A:0505+:9ABBBD29A>.59@D.505&9&5&)090&5+:?<>&)5&58+589@D50<?<<B@D>&&)0<&)",
            header=self.header,
        )
        
        pileup = Pileup(Range("1", 899721, 900192))
        pileup.add_read(read, ref_seq=ref_seq)

        for read_idx, pileup_idx in enumerate(range(148, 287)): # Matched bases
            base = pileup._columns[pileup_idx][0]
            if read.query_sequence[read_idx] == ref_seq[pileup_idx]:
                self.assertEqual(base.aligned, BaseAlignment.MATCH)
            else:
                self.assertEqual(base.aligned, BaseAlignment.MISMATCH)