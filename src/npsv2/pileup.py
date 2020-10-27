from .range import Range
import pysam


CIGAR_ADVANCE_PILEUP = frozenset(
    [pysam.CMATCH, pysam.CDEL, pysam.CREF_SKIP, pysam.CSOFT_CLIP, pysam.CEQUAL, pysam.CDIFF,]
)

CIGAR_ALIGNED_BASE = frozenset([pysam.CMATCH, pysam.CEQUAL, pysam.CDIFF,])


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


class Fragment(object):
    def __init__(self, read):
        # TODO: Allow for supplementary alignments, if read not already present
        assert Fragment.is_primary(read)
        self.read1 = read
        self.read2 = None

    @staticmethod
    def is_primary(read):
        return not read.is_supplementary and not read.is_secondary

    @property
    def is_properly_paired(self):
        return (
            self.read1 is not None
            and self.read2 is not None
            and self.read1.is_paired
            and self.read1.is_reverse != self.read2.is_reverse
        )

    @property
    def fragment_length(self):
        assert self.read1.template_length == (self.read2.reference_end - self.read1.reference_start)
        return self.read1.template_length
       

    def add_read(self, read):
        assert Fragment.is_primary(read)
        assert read.is_paired and read.query_name == self.read1.query_name
        assert read.__hash__() != self.read1.__hash__()

        self.read2 = read
        if self.read2.reference_start < self.read1.reference_start:
            self.read1, self.read2 = self.read2, self.read1

    def fragment_straddles(self, left_region: Range, right_region: Range, min_aligned=3):
        # TODO: Allow fragments with just one read
        if not self.is_properly_paired:
            return False

        return (
            left_region.get_overlap(self.read1) >= min_aligned and right_region.get_overlap(self.read2) >= min_aligned
        )


class FragmentTracker(object):
    def __init__(self):
        self._fragments = {}

    def __iter__(self):
        return iter(self._fragments.values())

    def __len__(self):
        return len(self._fragments)

    def add_read(self, read):
        name = read.query_name
        if name in self._fragments:
            self._fragments[name].add_read(read)
        else:
            self._fragments[name] = Fragment(read)
