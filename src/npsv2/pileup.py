from .range import Range
import pysam


CIGAR_ADVANCE_PILEUP = frozenset(
    [pysam.CMATCH, pysam.CDEL, pysam.CREF_SKIP, pysam.CSOFT_CLIP, pysam.CEQUAL, pysam.CDIFF,]
)

CIGAR_ALIGNED_BASE = frozenset([
    pysam.CMATCH,
    pysam.CEQUAL,
    pysam.CDIFF,
])

def _read_region(read, cigar):
    first_op, first_len = cigar[0]
    if first_op == pysam.CSOFT_CLIP:
        read_start = read.reference_start - first_len 
    else:
        read_start = read.reference_start
    aligned_length = sum(length for op, length in cigar if op in CIGAR_ADVANCE_PILEUP)
    return read_start, read_start + aligned_length


class PileupColumn(object):
    def __init__(self):
        self.aligned_bases = 0
        self.soft_clipped_bases = 0

    @property
    def total_bases(self):
        return self.aligned_bases + self.soft_clipped_bases

    def apply_cigar(self, cigar_op):
        if cigar_op == pysam.CSOFT_CLIP:
            self.soft_clipped_bases += 1
        elif cigar_op in CIGAR_ALIGNED_BASE:
            self.aligned_bases += 1


class Pileup(object):
    def __init__(self, reference_region: Range):
        self._region = reference_region
        self._columns = [PileupColumn() for _ in range(self._region.length)]
        
        self.read_count = 0

    def __len__(self):
        return len(self._columns)

    def __getitem__(self, sliced):
        return self._columns[sliced]

    def add_read(self, read: pysam.AlignedSegment):
        if read.reference_name != self._region.contig:
            return False

        # Determine read region on the reference, including and leading or trailing soft clipped regions
        cigar = read.cigartuples
        if not cigar:
            return False

        read_start, read_end = _read_region(read, cigar)
        if read_start >= self._region.end or read_end <= self._region.start:
            return False
        
        pileup_start = read_start - self._region.start
        for cigar_op, cigar_len in cigar:
            if cigar_op in CIGAR_ADVANCE_PILEUP:
                for col in range(max(pileup_start, 0), min(pileup_start + cigar_len, len(self._columns))):
                    self._columns[col].apply_cigar(cigar_op)
                pileup_start += cigar_len

        self.read_count += 1
        return True
