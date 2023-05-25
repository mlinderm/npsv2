import logging, math, os, re, tempfile, textwrap, typing
import pysam
from .range import Range
from . import npsv2_pb2
from .utilities.sequence import as_scalar

_VALID_SNV_ALLELES = frozenset("ACGTNacgtn")
_VALID_BASES_RE = re.compile(r"[ACGTN]+")

_ALLELES_TO_IUPAC = {
    frozenset("A"): "A",
    frozenset("C"): "C",
    frozenset("G"): "G",
    frozenset("T"): "T",
    frozenset("AG"): "R",
    frozenset("CT"): "Y",
    frozenset("GC"): "S",
    frozenset("AT"): "W",
    frozenset("GT"): "K",
    frozenset("AC"): "M",
    frozenset("CGT"): "B",
    frozenset("AGT"): "D",
    frozenset("ACT"): "H",
    frozenset("ACG"): "V",
    frozenset("ACGT"): "N",
}

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
    "N": frozenset("ACGT"),
} 



def _snv_alleles(record: pysam.VariantRecord) -> typing.FrozenSet:
    alleles = []
    for allele in record.alleles:
        if len(allele) != 1 or allele not in _VALID_SNV_ALLELES:
            return frozenset()
        alleles.append(allele.upper())
    return frozenset(alleles)


def _reference_sequence(reference_fasta: str, region: Range, snv_vcf_path: str = None) -> str:
    with pysam.FastaFile(reference_fasta) as ref_fasta:
        # Make sure reference sequence is all upper case
        ref_seq = ref_fasta.fetch(reference=region.contig, start=region.start, end=region.end).upper()

    # If SNV VCF is provided, modify the reference sequence with IUPAC codes
    if snv_vcf_path is None:
        return ref_seq

    with pysam.VariantFile(snv_vcf_path) as vcf_file:
        # TODO: We currently drop all samples and use all reported alleles, in the future only use alleles present
        # in a specified sample
        vcf_file.subset_samples([])  # Drop all samples
        for record in vcf_file.fetch(**region.pysam_fetch):
            alleles = _snv_alleles(record)
            if len(alleles) == 0:  # TODO: Add additional filtering criteria?
                continue
            ref_seq_index = record.start - region.start

            # The ref_seq might already be converted to IUPAC, make sure the VCF reference is compatible with those alleles
            ref_alleles = _IUPAC_TO_ALLELES[ref_seq[ref_seq_index]]
            assert record.ref in ref_alleles, f"VCF REF {record.contig}:{record.pos}{record.ref} not compatible with {ref_seq[ref_seq_index]}"
            alleles |= ref_alleles
             
            # Replace reference base with single letter IUPAC code
            ref_seq = ref_seq[:ref_seq_index] + _ALLELES_TO_IUPAC.get(alleles, "N") + ref_seq[ref_seq_index + 1 :]

    assert len(ref_seq) == region.length
    return ref_seq


def _phased_reference_sequence(reference_fasta: str, region: Range, snv_vcf_path: str, sample: str) -> typing.Tuple[str,str]:
    with pysam.FastaFile(reference_fasta) as ref_fasta:
        # Make sure reference sequence is all upper case
        ref_seq = ref_fasta.fetch(reference=region.contig, start=region.start, end=region.end).upper()
    
    # If SNV VCF is provided, modify the reference sequence with phased SNVs
    if snv_vcf_path is None:
        return (ref_seq, ref_seq)
    
    seq0, seq1 = ref_seq, ref_seq
    with pysam.VariantFile(snv_vcf_path) as vcf_file:
        vcf_file.subset_samples([sample])  # Drop all but relevant sample
        phase_sets = set()
        for record in vcf_file.fetch(**region.pysam_fetch):
            # Only consider SNVs
            if not all(len(allele) == 1 for allele in record.alleles):
                continue
            
            # Only apply phased genotypes, and require single phase set
            call = record.samples[0]
            if not call.phased:
                continue
            
            # TODO: Do we want this restriction?
            ps = call.get("PS", None)
            if ps is not None:
                phase_sets.add(ps)
            #if len(phase_sets) > 1:
            #    return (ref_seq, ref_seq)

            ref_seq_index = record.start - region.start
            alleles = call.alleles
            assert len(alleles) == 2
            seq0 = seq0[:ref_seq_index] + alleles[0] + seq0[ref_seq_index + 1 :]
            seq1 = seq1[:ref_seq_index] + alleles[1] + seq1[ref_seq_index + 1 :]

    return (seq0, seq1)


def allele_indices_to_ac(indices, alleles: typing.AbstractSet[int] = {1}) -> typing.Optional[int]:
    count = 0
    for idx in indices:
        if idx == -1:
            return None
        elif idx in alleles:
            count += 1
    return count


def _has_symbolic_allele(record):
    for alt in record.alts:
        if alt.startswith("<") or alt.endswith(">"):
            return True
    return False


class Variant(object):
    def __init__(self, record: pysam.VariantRecord):
        """Initialize Variant object from pysam.VariantRecord

        Args:
            record (pysam.VariantRecord): Underlying VariantRecord
        """
        self._record = record

        self._sequence_resolved = not _has_symbolic_allele(record)
        if self._sequence_resolved:
            self._padding = len(os.path.commonprefix(record.alleles))
            self._right_padding = [
                len(os.path.commonprefix([record.ref[self._padding :][::-1], a[self._padding :][::-1]]))
                for a in record.alts
            ]
        else:
            assert len(record.alts) == 1, "Multiallelic symbolic variants not currently supported"
            self._padding = 1
            self._right_padding = [0] * len(record.alts)
        if self._padding > 1:
            logging.warning("Variant has more than expected number of padding bases, is the VCF normalized?")

    @classmethod
    def from_pysam(cls, record: pysam.VariantRecord) -> "Variant":
        """Factory method for creating appropriate Variant objects"""
        if "SVTYPE" in record.info:
            svtype = as_scalar(record.info["SVTYPE"])
        elif _has_symbolic_allele(record):
            svtype = os.path.commonprefix([a.strip("<>") for a in record.alts])
        else:
            svlen = tuple(len(a) - len(record.ref) for a in record.alts)
            if all(map(lambda l: l < 0, svlen)):
                svtype = "DEL"
            elif all(map(lambda l: l > 0, svlen)):
                svtype = "INS"
            elif all(map(lambda l: l == 0, svlen)):
                svtype = "SUB"
            else:
                svtype = "DEL"
                #raise ValueError("Inconsistent variant types")
        if svtype.startswith("DEL"):
            return DeletionVariant(record)
        elif svtype.startswith("INS"):
            return InsertionVariant(record)
        elif svtype == "SUB":
            return SubstitutionVariant(record)
        else:
            raise ValueError(f"Variant kind {svtype} not supported")

    @property
    def allele_indices(self):
        return range(1 + self.num_alt)

    @property
    def alt_allele_indices(self):
        return range(1, 1 + self.num_alt)

    @property
    def name(self):
        return f"{self.contig}_{self.start + 1}_{self.end}_{self.type}"

    @property
    def type(self) -> str:
        return npsv2_pb2.StructuralVariant.Type.Name(self._svtype_as_proto)

    @property
    def is_deletion(self):
        return False

    @property
    def is_insertion(self):
        return False

    @property
    def is_substitution(self):
        return False

    @property
    def is_SNV(self):
        return False

    @property
    def is_MNV(self):
        return False

    @property
    def contig(self):
        return self._record.contig

    @property
    def start(self):
        return self._record.start

    @property
    def end(self):
        return self._record.stop

    @property
    def num_alt(self):
        # TODO: Exclude alleles?
        return len(self._record.alts)

    def is_biallelic(self):
        # TODO: Exclude alleles
        return self.num_alt == 1

    def length_change(self, allele=1):
        svlen = self._record.info.get("SVLEN", None)
        if svlen is None:
            svlen = tuple(self.alt_length(i + 1) - self.ref_length for i in range(self.num_alt))
        elif isinstance(svlen, int):
            svlen = (svlen,)  # If SVLEN is Number=1, convert to sequence
        return svlen if allele is None else svlen[allele - 1]

    @property
    def ref_length(self):
        """Length of reference allele including any padding bases"""
        raise NotImplementedError()

    def alt_length(self, allele=1):
        """Length of alternate allele including any padding bases"""
        raise NotImplementedError()

    @property
    def reference_region(self) -> Range:
        """Returns changed region of the reference genome, excluding any padding bases"""
        return Range(self.contig, self.start + self._padding, self.end)

    def left_flank_region(self, left_flank, right_flank=0, allele=1):
        start = self.start + self._padding
        return Range(self.contig, start - left_flank, start + right_flank)

    def right_flank_region(self, right_flank, left_flank=0, allele=1):
        end = self.end - self._right_padding[allele - 1]
        return Range(self.contig, end - left_flank, end + right_flank)

    def ref_breakpoints(self, flank, allele=1, contig=None):
        if contig is None:
            contig = self.contig
        event_end = flank + self.ref_length - self._padding - self._right_padding[allele - 1]
        return (
            Range(contig, flank - 1, flank + 1),
            (Range(contig, event_end - 1, event_end + 1) if event_end > flank else None),
        )

    def alt_breakpoints(self, flank, allele=1, contig=None):
        if contig is None:
            contig = self.contig
        event_end = flank + self.alt_length(allele) - self._padding - self._right_padding[allele - 1]
        return (
            Range(contig, flank - 1, flank + 1),
            (Range(contig, event_end - 1, event_end + 1) if event_end > flank else None),
        )

    def genotype_indices(self, index_or_id):
        call = self._record.samples[index_or_id]
        return call.allele_indices if call else None

    def genotype_allele_count(self, index_or_id, alleles: typing.AbstractSet[int] = None):
        indices = self.genotype_indices(index_or_id)
        if indices is None:
            return None
        if alleles is None:
            alleles = frozenset(self.alt_allele_indices)
        count = 0
        for idx in indices:
            if idx == -1:
                return None
            elif idx in alleles:
                count += 1
        return count

    def _alt_seq(self, ref_seq, flank, allele=1, right_flank=None):
        raise NotImplementedError()


    def _contig_names(self, region: Range, ref_contig=None, alt_contig=None):
        if ref_contig is None:
            ref_contig = str(region).replace(":", "_").replace("-", "_")

        if alt_contig is None:
            alt_contig = [f"{ref_contig}_{i}_alt" for i in self.alt_allele_indices]
        elif isinstance(alt_contig, str) and self.num_alt == 1:
            alt_contig = [alt_contig]
        if len(alt_contig) != self.num_alt:
            raise ValueError("There must be one alt_contig name for each alternate allele")

        return ref_contig, alt_contig


    def synth_fasta(
        self,
        reference_fasta,
        alleles=(0, 1),
        flank=1,
        ref_contig=None,
        alt_contig=None,
        line_width=60,
        dir=None,
        snv_vcf_path: str = None,
        index_mode=False,
    ):
        region = self.reference_region.expand(flank)
        ref_contig, alt_contig = self._contig_names(region, ref_contig=ref_contig, alt_contig=alt_contig)
        ref_seq = _reference_sequence(reference_fasta, region, snv_vcf_path=snv_vcf_path)
  
        unique_alleles = set(alleles)

        # Write out FASTA
        allele_fasta = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta", dir=dir)
       
        # Always write out REF header, but only write sequence if reference allele is present or we are generating sequences
        # for the alignment index
        print(f">{ref_contig}", file=allele_fasta)
        if index_mode or 0 in unique_alleles:
            for line in textwrap.wrap(ref_seq, width=line_width):
                print(line, file=allele_fasta)

        for allele_idx, contig in zip(self.alt_allele_indices, alt_contig):
            if allele_idx in unique_alleles:
                print(f">{contig}", file=allele_fasta)
                alt_seq = self._alt_seq(ref_seq, flank, allele=allele_idx)
                for line in textwrap.wrap(alt_seq, width=line_width):
                    print(line, file=allele_fasta)
            elif not index_mode:
                # Only write header without sequence if not in index mode (i.e. simulation mode)
                print(f">{contig}", file=allele_fasta)

        # Flatten alt_contig if only a single alternate allele
        return allele_fasta.name, ref_contig, (alt_contig[0] if self.num_alt == 1 else alt_contig)


    def phase_synth_fasta(
        self,
        reference_fasta,
        snv_vcf_path: str,
        sample: str,
        alleles=(0, 1),
        flank=1,
        ref_contig=None,
        alt_contig=None,
        line_width=60,
        dir=None,
    ):
        assert len(alleles) == 2
        region = self.reference_region.expand(flank)
        ref_contig, alt_contig = self._contig_names(region, ref_contig=ref_contig, alt_contig=alt_contig)
        seqs = _phased_reference_sequence(reference_fasta, region, snv_vcf_path=snv_vcf_path, sample=sample)

        # Write out FASTA, duplicating contigs with different haplotypes as appropriate
        allele_fasta = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta", dir=dir)

        next_seq = 0
        fasta_contigs = []
        for allele_idx, contig in zip(self.allele_indices, [ref_contig] + alt_contig):
            allele_count = alleles.count(allele_idx)
            
            # Make sure to generate the sequence header, even without sequence
            if allele_count == 0:
                print(f">{contig}", file=allele_fasta)
            for i in range(allele_count):
                fasta_contigs.append(f">{contig}_hap{next_seq}")
                print(fasta_contigs[-1], file=allele_fasta)
                seq = seqs[next_seq] if allele_idx == 0 else self._alt_seq(seqs[next_seq], flank, allele=allele_idx)
                for line in textwrap.wrap(seq, width=line_width):
                    print(line, file=allele_fasta)
                next_seq += 1

        assert len(fasta_contigs) == 2
        return allele_fasta.name, fasta_contigs[0], fasta_contigs[1]


    @property
    def _svtype_as_proto(self):
        raise NotImplementedError()

    def as_proto(self):
        sv = npsv2_pb2.StructuralVariant()
        sv.contig = self.contig
        sv.start = self.start
        sv.end = self.end
        sv.svlen.extend(self.length_change(allele=None))
        sv.svtype = self._svtype_as_proto
        return sv


class DeletionVariant(Variant):
    def __init__(self, record):
        Variant.__init__(self, record)

    @property
    def is_deletion(self):
        return True

    @property
    def ref_length(self):
        return self.end - self.start

    def alt_length(self, allele=1):
        assert allele >= 1
        if self._sequence_resolved:
            alt_allele = self._record.alleles[allele]
            return len(alt_allele)
        else:
            return 1

    def _alt_seq(self, ref_seq, flank, allele=1, right_flank = None):
        if right_flank is None:
            right_flank = flank
        if self._sequence_resolved:
            alt_allele = self._record.alleles[allele].upper()
            assert _VALID_BASES_RE.fullmatch(alt_allele), "Unexpected base in sequence resolved allele"
            return ref_seq[:flank] + alt_allele[self._padding :] + ref_seq[len(ref_seq)-right_flank:]
        else:
            # TODO: This may not be valid for multi-allelic variants
            assert self.num_alt == 1
            return ref_seq[:flank] + ref_seq[-right_flank:]

    def window_regions(self, window_size: int, flank_windows: int, window_interior=True):
        # We anchor windows on the breakpoints
        assert window_size % 2 == 0, "Window size must be evenly split in half"
        breakpoint_flank = window_size // 2

        # How many interior windows are needed on each side?
        interior = self.reference_region
        if window_interior:
            interior_windows = (max(interior.length - window_size, 0) // window_size) // 2
        else:
            interior_windows = flank_windows

        left_region = self.left_flank_region(
            window_size * flank_windows + breakpoint_flank, breakpoint_flank + interior_windows * window_size
        )
        right_region = self.right_flank_region(
            window_size * flank_windows + breakpoint_flank, breakpoint_flank + interior_windows * window_size
        )
        if left_region.end >= right_region.start or not window_interior:
            # Regions abut or overlap, or we are not specifically tiling the interior
            return left_region.window(window_size) + right_region.window(window_size)
        else:
            assert (
                right_region.start - left_region.end
            ) <= window_size, "Should only be one potentially overlapping 'center' region"
            center_region = interior.center.expand(breakpoint_flank)
            return left_region.window(window_size) + [center_region] + right_region.window(window_size)


    @property
    def _svtype_as_proto(self):
        return npsv2_pb2.StructuralVariant.Type.DEL


class _SequenceResolvedVariant(Variant):
    def __init__(self, record):
        Variant.__init__(self, record)
        if not self._sequence_resolved:
            raise ValueError("Symbolic inserstions are not supported")

    @property
    def is_insertion(self):
        return True

    @property
    def ref_length(self):
        return len(self._record.alleles[0])

    def alt_length(self, allele=1):
        assert allele >= 1
        assert self._sequence_resolved
        alt_allele = self._record.alleles[allele]
        return len(alt_allele)

    def _alt_seq(self, ref_seq, flank, allele=1):
        assert allele >= 1
        assert self._sequence_resolved

        alt_allele = self._record.alleles[allele].upper()
        assert _VALID_BASES_RE.fullmatch(alt_allele), "Unexpected base in sequence resolved allele"
        return ref_seq[:flank] + alt_allele[self._padding :] + ref_seq[-flank:]


class InsertionVariant(_SequenceResolvedVariant):
    @property
    def is_insertion(self):
        return True

    @property
    def _svtype_as_proto(self):
        return npsv2_pb2.StructuralVariant.Type.INS


class SubstitutionVariant(_SequenceResolvedVariant):
    @property
    def is_substitution(self):
        return True

    @property
    def is_SNV(self):
        return self.ref_length == 1

    @property
    def is_MNV(self):
        return self.ref_length > 1

    @property
    def _svtype_as_proto(self):
        return npsv2_pb2.StructuralVariant.Type.SUB

