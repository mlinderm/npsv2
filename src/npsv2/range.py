from collections import defaultdict
from pysam.libcutils import parse_region
from pysam import AlignedSegment
from intervaltree import Interval, IntervalTree

class Range(object):
    # Zero indexed start, 0-indexed exclusive end
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

    def __hash__(self):
        return hash((self.contig, self.start, self.end))

    @property
    def length(self):
        return self.end - self.start

    @property
    def pysam_fetch(self):
        return dict(contig=self.contig, start=self.start, stop=self.end)

    @property
    def center(self):
        start = self.start + self.length // 2
        return Range(self.contig, start, start)

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
        elif isinstance(has_region, Range):
            if self.contig != has_region.contig:
                return 0
            return max(0, min(self.end, has_region.end) - max(self.start, has_region.start))
        else:
            raise NotImplementedError()

    def union(self, other: "Range") -> "Range":
        if self.contig != other.contig:
            raise ValueError("Can't union Ranges with different contigs")
        return Range(self.contig, min(self.start, other.start), max(self.end, other.end))

    def intersection(self, other: "Range") -> "Range":
        if self.contig != other.contig or other.start >= self.end or self.start >= other.end:
            return Range("", 0, 0)
        else:
            return Range(self.contig, max(self.start, other.start), min(self.end, other.end))

    def window(self, size):
        assert self.length % size == 0
        return [Range(self.contig, s, s+size) for s in range(self.start, self.end, size)]

class RangeTree:
    def __init__(self):
        self._trees = defaultdict(IntervalTree)

    def add(self, region: Range, data):
        return self._trees[region.contig].addi(region.start, region.end, data)

    def overlap(self, region: Range):
        return self._trees[region.contig].overlap(region.start, region.end)

    def merge_overlaps(self, **kwargs):
        for tree in self._trees.values():
            tree.merge_overlaps(**kwargs)

    def values(self):
        for tree in self._trees.values():
            for _, _, data in tree.items():
                yield data
