from pysam.libcutils import parse_region
from pysam import AlignedSegment

class Range(object):
    def __init__(self, contig, start, end):
        self.contig = contig
        self.start = start
        self.end = end

    @classmethod
    def parse_literal(cls, region):
        contig, start, end = parse_region(region=region)
        return Range(contig, start, end)

    def __eq__(self, rhs):
        return isinstance(rhs, Range) and self.contig == rhs.contig and self.start == rhs.start and self.end == rhs.end

    def __str__(self):
        return f"{self.contig}:{self.start+1}-{self.end}"

    @property
    def length(self):
        return self.end - self.start

    # TODO: Add contig map
    def expand(self, left_or_both, right=None):
        if right is None:
            right = left_or_both
        new_start = max(self.start - left_or_both, 0)
        new_end = self.end + right
        return Range(self.contig, new_start, new_end)

    def get_overlap(self, read: AlignedSegment):
        if read.reference_name != self.contig:
            return 0
        return read.get_overlap(self.start, self.end)
