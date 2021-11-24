import itertools, logging, random, sys
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple
import pysam
from intervaltree import Interval
from .range import Range


CIGAR_ADVANCE_PILEUP = frozenset(
    [pysam.CMATCH, pysam.CDEL, pysam.CREF_SKIP, pysam.CSOFT_CLIP, pysam.CEQUAL, pysam.CDIFF,]
)

CIGAR_ADVANCE_READ = frozenset(
    [pysam.CMATCH, pysam.CINS, pysam.CSOFT_CLIP, pysam.CEQUAL, pysam.CDIFF,]
)

CIGAR_ALIGNED_BASE = frozenset([pysam.CMATCH, pysam.CEQUAL, pysam.CDIFF,])


class AlleleAssignment(Enum):
    AMB = -1
    REF = 0
    ALT = 1


class BaseAlignment(Enum):
    ALIGNED = 3
    MATCH=1
    MISMATCH=2
    SOFT_CLIP = 4
    INSERT = 8  # No bases, but within insert

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


_CIGAR_TO_BASE_ALIGNMENT = {
    pysam.CMATCH: BaseAlignment.ALIGNED,
    pysam.CEQUAL: BaseAlignment.MATCH,
    pysam.CDIFF: BaseAlignment.MISMATCH,
    pysam.CSOFT_CLIP: BaseAlignment.SOFT_CLIP,
}


class Strand(Enum):
    POSITIVE = 1
    NEGATIVE = -1


def _read_region(read, cigar):
    first_op, first_len = cigar[0]
    if first_op == pysam.CSOFT_CLIP:
        read_start = read.reference_start - first_len
    else:
        read_start = read.reference_start
    aligned_length = sum(length for op, length in cigar if op in CIGAR_ADVANCE_PILEUP)
    return Range(read.reference_name, read_start, read_start + aligned_length)


def _read_strand(read: pysam.AlignedSegment):
    return Strand.NEGATIVE if read.is_reverse else Strand.POSITIVE


def read_start(read):
    first_op, first_len = read.cigar[0]
    if first_op == pysam.CSOFT_CLIP:
        return read.reference_start - first_len
    else:
        return read.reference_start


_IUPAC_TO_ALLELES = {
    "A": frozenset("A"),
    "C": frozenset("C"), 
    "G": frozenset("G"), 
    "T": frozenset("T"), 
    "R": frozenset("AG"),
    "Y": frozenset("CT"),
    "S": frozenset("GC"),
    "W": frozenset("AT"),
    "K": frozenset("GT"),
    "M": frozenset("AC"),
    "B": frozenset("CGT"),
    "D": frozenset("AGT"),
    "H": frozenset("ACT"),
    "V": frozenset("ACG"),
    "N": frozenset("ACGTN"),
}

def _refine_cigar_op(base: str, ref: str):
    # Treat any of the IUPAC alleles as matching
    match = base.upper() in _IUPAC_TO_ALLELES[ref.upper()]
    return pysam.CEQUAL if match else pysam.CDIFF

def _refine_base_alignment(base: str, ref: str):
    # Treat any of the IUPAC alleles as matching
    match = base.upper() in _IUPAC_TO_ALLELES[ref.upper()]
    return BaseAlignment.MATCH if match else BaseAlignment.MISMATCH



@dataclass
class AlleleRealignment:
    ref_quality: int = None
    alt_quality: int = None
    allele: AlleleAssignment = AlleleAssignment.AMB
    # breakpoint: bool
    # quality: float = 0
    # normalized_score: float = 0


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
    def query_name(self):
        return self.read1.query_name

    @property
    def reads(self):
        return (self.read1, self.read2) if self.read1 and self.read2 else (self.read1,)

    @property
    def is_properly_paired(self):
        return (
            self.read1 is not None
            and self.read2 is not None
            and self.read1.is_paired
            and self.read1.is_reverse != self.read2.is_reverse
        )

    @property
    def fragment_region(self):
        assert self.is_properly_paired
        return Range(self.read1.reference_name, self.read1.reference_start, self.read2.reference_end)

    @property
    def fragment_start(self):
        return self.read1.reference_start

    @property
    def fragment_length(self):
        # TODO: Handle weird pairs
        # if self.read1.template_length != (self.read2.reference_end - self.read1.reference_start):
        #     print(self.read1)
        #     print(self.read2)
        # assert self.read1.template_length == (self.read2.reference_end - self.read1.reference_start)
        return self.read1.template_length

    @property
    def insert_region(self):
        assert self.is_properly_paired
        return Range(self.read1.reference_name, self.read1.reference_end, self.read2.reference_start)

    def add_read(self, read):
        assert Fragment.is_primary(read)
        assert read.is_paired and read.query_name == self.read1.query_name
        assert read != self.read1

        self.read2 = read
        if self.read2.reference_start < self.read1.reference_start:
            self.read1, self.read2 = self.read2, self.read1

    def fragment_straddles(self, left_region: Range, right_region: Range, min_aligned=3):
        assert left_region.contig == right_region.contig
        if left_region.length <= min_aligned or right_region.length <= min_aligned:
            print(left_region, right_region, file=sys.stderr)
        assert left_region.length > min_aligned and right_region.length > min_aligned

        # TODO: Allow fragments with just one read
        if not self.is_properly_paired:
            return False
        if self.read1.reference_name != left_region.contig:
            return False
        return (
            (left_region.start <= self.read1.reference_start < (left_region.end - min_aligned)) and 
            (right_region.start <= self.read2.reference_start < (right_region.end - min_aligned))
        )

        return (
            left_region.get_overlap(self.read1) >= min_aligned and right_region.get_overlap(self.read2) >= min_aligned
        )

    def fragment_overlaps(self, region: Range, min_overlap=3, read_overlap_only=False):
        if self.is_properly_paired:
            return self.fragment_region.get_overlap(region) >= min_overlap
        else:
            return self.reads_overlap(region, min_overlap)

    def reads_overlap(self, region: Range, min_overlap=3):
        return (self.read1 and (region.get_overlap(self.read1) >= min_overlap)) or (self.read2 and (region.get_overlap(self.read2) >= min_overlap))


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


class PileupBase(NamedTuple):
    read_start: int
    aligned: BaseAlignment
    mapq: int = None
    allele: AlleleRealignment = AlleleRealignment() #AlleleAssignment.AMB, False, 0, 0)
    ref_zscore: float = None
    alt_zscore: float = None
    strand: Strand = None
    baseq: int = None


class PileupColumn:
    def __init__(self):
        self._bases = []

    def __len__(self):
        return len(self._bases)

    def __getitem__(self, sliced):
        return self._bases[sliced]

    def _aligned_count(self, kind):
        return sum((base.aligned == kind for base in self._bases))

    @property
    def aligned_bases(self):
        return self._aligned_count(BaseAlignment.ALIGNED)

    @property
    def soft_clipped_bases(self):
        return self._aligned_count(BaseAlignment.SOFT_CLIP)

    def add_base(self, read_start, cigar_op_or_alignment, **attributes):
        if isinstance(cigar_op_or_alignment, BaseAlignment):
            self._bases.append(PileupBase(read_start, cigar_op_or_alignment, **attributes))
        else:
            align = _CIGAR_TO_BASE_ALIGNMENT.get(cigar_op_or_alignment, None)
            if align:
                self._bases.append(PileupBase(read_start, align, **attributes))

    def ordered_bases(self, max_bases=None, order="read_start"):
        # Return bases sorted by specified field, randomly downsampling if needed
        if max_bases is not None and len(self._bases) > max_bases:
            bases = random.sample(self._bases, k=max_bases)
        else:
            bases = self._bases
        sort_idx = PileupBase._fields.index(order)
        return sorted(bases, key=lambda base: base[sort_idx])

class Pileup:
    def __init__(self, reference_region: Range):
        self._region = reference_region
        self._columns = [PileupColumn() for _ in range(self._region.length)]

    def __len__(self):
        return len(self._columns)

    def __getitem__(self, sliced):
        return self._columns[sliced]

    def __iter__(self):
        return iter(self._columns)

    def region_columns(self, region: Range):
        if region.contig != self._region.contig:
            return []

        pileup_start = region.start - self._region.start
        next_slice = slice(max(pileup_start, 0), min(pileup_start + region.length, len(self._columns)))
        if next_slice.stop > next_slice.start:
            return [next_slice]
        else:
            return []

    def read_columns(self, read: pysam.AlignedSegment):
        if read.reference_name != self._region.contig:
            return None, []
        
        cigar = read.cigartuples
        if not cigar:
            return None, []
        
        first_op, first_len = cigar[0]
        if first_op == pysam.CSOFT_CLIP:
            read_start = read.reference_start - first_len
        else:
            read_start = read.reference_start
        
        if read_start >= self._region.end:
            return read_start, []
        
        slices = []
        pileup_index = read_start - self._region.start
        read_index = 0
        for cigar_op, cigar_len in cigar:
            if cigar_op in CIGAR_ADVANCE_PILEUP:
                next_slice = slice(max(pileup_index, 0), min(pileup_index + cigar_len, len(self._columns)))
                next_slice_len = next_slice.stop - next_slice.start
                if next_slice_len > 0 and cigar_op in CIGAR_ADVANCE_READ:
                    read_slice_start = read_index + next_slice.start - pileup_index
                    slices.append((next_slice, cigar_op, slice(read_slice_start, read_slice_start + next_slice_len)))
                elif next_slice_len > 0:
                    slices.append((next_slice, cigar_op, slice(read_index, read_index)))
                pileup_index += cigar_len
            if cigar_op in CIGAR_ADVANCE_READ:
                read_index += cigar_len

        return read_start, slices

    def add_read(self, read: pysam.AlignedSegment, ref_seq: str = None, **attributes):
        assert ref_seq is None or len(ref_seq) == len(self._columns)

        read_start, slices = self.read_columns(read)
        for col_slice, cigar_op, read_slice in slices:
            if ref_seq and cigar_op == pysam.CMATCH:
                # Refine CMATCH to CEQUAL or CDIFF so that we can render match/mismatch
                revised_cigar_op = map(_refine_cigar_op, read.query_sequence[read_slice], ref_seq[col_slice])
            else:
                revised_cigar_op = itertools.repeat(cigar_op, len(range(*col_slice.indices(len(self)))))
            for column, baseq, base_cigar_op in zip(self._columns[col_slice], read.query_qualities[read_slice], revised_cigar_op):
                column.add_base(read_start, base_cigar_op, mapq=read.mapping_quality, strand=_read_strand(read), baseq=baseq, **attributes)

        

    def add_insert(self, insert_region: Range, **attributes):
        # Exclude the allele for the insert bases (even though the fragment may be assigned to an allele)
        attributes.update({ "allele": AlleleRealignment(None, False, 0, 0) })
        overlap = self._region.intersection(insert_region)
        for column in self._columns[(overlap.start - self._region.start):(overlap.end - self._region.start)]:
            column.add_base(overlap.start, BaseAlignment.INSERT, **attributes)

    def add_fragment(self, fragment: Fragment, add_insert=False, ref_seq: str = None, **attributes):  
        if fragment.read1:
            self.add_read(fragment.read1, ref_seq=ref_seq, **attributes)
        if fragment.read2:
            self.add_read(fragment.read2, ref_seq=ref_seq, **attributes)
        if add_insert and fragment.is_properly_paired:
            self.add_insert(fragment.insert_region, **attributes)


def _read_columns(region: Range, read: pysam.AlignedSegment):
    if read.reference_name != region.contig:
        return []
    
    cigar = read.cigartuples
    if not cigar:
        return []
    
    first_op, first_len = cigar[0]
    if first_op == pysam.CSOFT_CLIP:
        read_start = read.reference_start - first_len
    else:
        read_start = read.reference_start
    
    if read_start >= region.end:
        return []
    
    slices = []
    pileup_index = read_start - region.start
    pileup_end = region.length
    read_index = 0
    for cigar_op, cigar_len in cigar:
        if cigar_op in CIGAR_ADVANCE_PILEUP:
            next_slice = slice(max(pileup_index, 0), min(pileup_index + cigar_len, pileup_end))
            if next_slice.stop > next_slice.start:
                if cigar_op in CIGAR_ADVANCE_READ:
                    read_slice_start = read_index + next_slice.start - pileup_index
                    read_slice_end = read_slice_start + next_slice.stop - next_slice.start
                    slices.append((next_slice, cigar_op, slice(read_slice_start, read_slice_end)))
                else:
                    slices.append((next_slice, cigar_op, slice(read_index, read_index)))
            pileup_index += cigar_len
        if cigar_op in CIGAR_ADVANCE_READ:
            read_index += cigar_len
    return slices


def _region_columns(region: Range, render_region: Range):
    if region.contig != render_region.contig:
        return []
    
    region_start = render_region.start
    if region_start >= region.end:
        return []
    
    pileup_start = region_start - region.start
    next_slice = slice(max(pileup_start, 0), min(pileup_start + render_region.length, region.length))
    return [next_slice]


# TODO: Consolidate these Pileup classes with a common base class
class PileupRead:
    def __init__(self, read, allele: AlleleAssignment, ref_zscore: float, alt_zscore: float):
        self._read = read
        self.region = _read_region(read, read.cigartuples)
        self.allele = allele
        self.ref_zscore = ref_zscore
        self.alt_zscore = alt_zscore

    @property
    def read_start(self):
        return self.region.start

    @property
    def mapq(self):
        return self._read.mapping_quality

    @property
    def strand(self):
        return Strand.NEGATIVE if self._read.is_reverse else Strand.POSITIVE

    def baseq(self, read_slice):
        return self._read.query_qualities[read_slice]


class PileupInsert:
    def __init__(self, region: Range, allele: AlleleAssignment, ref_zscore: float, alt_zscore: float):
        self.region = region
        self.allele = allele
        self.ref_zscore = ref_zscore
        self.alt_zscore = alt_zscore
        self.mapq = None
        self.strand = None

    @property
    def read_start(self):
        return self.region.start

    def baseq(self, read_slice):
        return None


class ReadPileup:
    def __init__(self, reference_regions):
        if isinstance(reference_regions, Range):
            reference_regions = [reference_regions]
        self._reads = { r:[] for r in reference_regions }

    def region_columns(self, region: Range, region_item: Range):
        return _region_columns(region, region_item)

    def read_columns(self, region: Range, pileup_item, ref_seq: str = None):
        if isinstance(pileup_item, PileupRead):
            assert ref_seq is None or region.length == len(ref_seq)
            for col_slice, cigar_op, read_slice in _read_columns(region, pileup_item._read):
                align = _CIGAR_TO_BASE_ALIGNMENT.get(cigar_op, None)
                if align is None:
                    continue
                elif ref_seq and cigar_op == pysam.CMATCH:
                    # Refine CMATCH to CEQUAL or CDIFF so that we can render match/mismatch
                    align = list(map(_refine_base_alignment, pileup_item._read.query_sequence[read_slice], ref_seq[col_slice]))
                yield (col_slice, align, read_slice)
        elif isinstance(pileup_item, PileupInsert):
            for col_slice in _region_columns(region, pileup_item.region):
                yield (col_slice, BaseAlignment.INSERT, slice(0, col_slice.stop - col_slice.start))

    def overlapping_reads(self, region: Range, max_reads: int=None):
        reads = self._reads[region]
        if len(reads) > max_reads:
            reads = random.sample(reads, k=max_reads)
        return sorted(reads, key=lambda read: read.read_start)

    def add_read(self, read: pysam.AlignedSegment, **attributes):
        pileup_read = PileupRead(read, **attributes)
        for region, reads in self._reads.items():
            if region.get_overlap(pileup_read.region) > 0:
                reads.append(pileup_read)

    def add_insert(self, insert_region: Range, **attributes):
        # Exclude the allele for the insert bases (even though the fragment may be assigned to an allele)
        attributes.update({ "allele": AlleleRealignment() }) #None, False, 0, 0) })
        for region, reads in self._reads.items():
            overlap = region.intersection(insert_region)
            if overlap.length > 0:
                reads.append(PileupInsert(overlap, **attributes))

    def add_fragment(self, fragment: Fragment, add_insert=False, ref_seq: str=None, **attributes):
        if fragment.read1:
            self.add_read(fragment.read1, **attributes)
        if fragment.read2:
            self.add_read(fragment.read2, **attributes)
        if add_insert and fragment.is_properly_paired:
            self.add_insert(fragment.insert_region, **attributes)


class PileupFragment:
    def __init__(self, fragment, allele: AlleleRealignment, ref_zscore: float, alt_zscore: float):
        self._fragment = fragment
        
        # TODO: Handle soft-clip starts. Get a more robust start
        self.region = fragment.fragment_region
        
        self.allele = allele
        self.ref_zscore = ref_zscore
        self.alt_zscore = alt_zscore

    @property
    def reads(self):
        return self._fragment.reads

    @property
    def start(self):
        return self.region.start

    @property
    def fragment_region(self):
        return self._fragment.fragment_region


class FragmentPileup:
    def __init__(self, reference_regions): 
        self._fragments = { r:[] for r in reference_regions }


    def overlapping_fragments(self, region: Range, max_length: int=None):
        fragments = self._fragments[region]
        if len(fragments) > max_length:
            fragments = random.sample(fragments, k=max_length)
        return sorted(fragments, key=lambda fragment: fragment.start)


    def read_columns(self, region: Range, pileup_item, ref_seq: str = None):
        if isinstance(pileup_item, pysam.AlignedSegment):
            assert ref_seq is None or region.length == len(ref_seq)
            for col_slice, cigar_op, read_slice in _read_columns(region, pileup_item):
                align = _CIGAR_TO_BASE_ALIGNMENT.get(cigar_op, None)
                if align is None:
                    continue
                elif ref_seq and cigar_op == pysam.CMATCH:
                    # Refine CMATCH to CEQUAL or CDIFF so that we can render match/mismatch
                    align = list(map(_refine_base_alignment, pileup_item._read.query_sequence[read_slice], ref_seq[col_slice]))
                yield (col_slice, align, read_slice)
        elif isinstance(pileup_item, PileupInsert):
            for col_slice in _region_columns(region, pileup_item.region):
                yield (col_slice, BaseAlignment.INSERT, slice(0, col_slice.stop - col_slice.start))    


    def fragment_columns(self, region: Range, fragment: PileupFragment):
        return _region_columns(region, fragment.fragment_region)


    def add_fragment(self, fragment: Fragment, **attributes):
        pileup_fragment = PileupFragment(fragment, **attributes)
        for region, fragments in self._fragments.items():
            # Fragment is relevant if any of the reads overlap the region
            if fragment.reads_overlap(region, min_overlap=1):
                fragments.append(pileup_fragment)