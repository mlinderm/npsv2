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

    @property
    def pysam_fetch(self):
        return dict(contig=self.contig, start=self.start, stop=self.end)

    def contains(self, point: int):
        return self.start <= point < self.end

    # TODO: Add contig map
    def expand(self, left_or_both, right=None):
        if right is None:
            right = left_or_both
        new_start = max(self.start - left_or_both, 0)
        new_end = self.end + right
        return Range(self.contig, new_start, new_end)

    def get_overlap(self, has_region):
        # Tried @singledispatchmethod, but couldn't make it work for Range inputs
        if isinstance(has_region, AlignedSegment):
            if has_region.reference_name != self.contig:
                return 0
            return has_region.get_overlap(self.start, self.end)
        elif isinstance(has_region, "Range"):
            if self.contig != other.contig:
                return 0
            return max(0, min(self.end, other.end) - max(self.start, other.start))
        else:
            raise NotImplementedError()