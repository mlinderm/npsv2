import logging, math, os, tempfile, textwrap
import pysam
from .range import Range
from . import npsv2_pb2

def _reference_sequence(reference_fasta, region: Range) -> str:
    with pysam.FastaFile(reference_fasta) as ref_fasta:
        return ref_fasta.fetch(reference=region.contig, start=region.start, end=region.end)

class Variant(object):
    def __init__(self, record):
        self._record = record
        assert len(record.alts) == 1, "Multiple alternates are not supported"
        
        ref_allele = record.ref
        alt_allele = record.alts[0]
        self._sequence_resolved = not alt_allele.startswith("<")
        if self._sequence_resolved:
            padding_string = os.path.commonprefix([ref_allele, alt_allele])
            self._padding = len(padding_string)
        else:
            self._padding = 1
        if self._padding > 1:
            logging.warning("Variant has more than expected number of padding bases, if VCF normalized?")

    @classmethod
    def from_pysam(cls, record):
        # TODO: Actually detect SV kind
        assert record.info["SVTYPE"] == "DEL"
        return DeletionVariant(record)

    @property
    def is_deletion(self):
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

    def is_biallelic(self):
        # TODO: Exclude alleles
        return len(self._record.alts) == 1

    def length_change(self):
        svlen = self._record.info["SVLEN"]
        if isinstance(svlen, int):
            return svlen
        else:
            return svlen[0]

    @property
    def ref_length(self):
        """Length of reference allele including any padding bases"""
        raise NotImplementedError()

    @property
    def alt_length(self):
        """Length of alternate allele including any padding bases"""
        raise NotImplementedError()

    @property
    def reference_region(self):
        return Range(self.contig, self.start + self._padding, self.end)

    def left_flank_region(self, left_flank, right_flank=0):
        return Range(self.contig, self.start + self._padding - left_flank, self.start + self._padding + right_flank)
    
    def right_flank_region(self, right_flank, left_flank=0):
        return Range(self.contig, self.end - left_flank, self.end + right_flank)

    def ref_breakpoints(self, flank, contig=None):
        if contig is None:
            contig = self.contig
        event_end = flank + self.ref_length - self._padding
        return Range(contig, flank-1, flank+1), (Range(contig, event_end-1, event_end+1) if event_end > flank else None)

    def alt_breakpoints(self, flank, contig=None):
        if contig is None:
            contig = self.contig
        event_end = flank + self.alt_length - self._padding
        return Range(contig, flank-1, flank+1), (Range(contig, event_end-1, event_end+1) if event_end > flank else None)

    def genotype_indices(self, index_or_id):
        call = self._record.samples[index_or_id]
        return call.allele_indices if call else None

    def _alt_seq(self, flank, ref_seq):
        raise NotImplementedError()  

    def synth_fasta(
        self,
        reference_fasta,
        ac=1,
        flank=1,
        ref_contig=None,
        alt_contig=None,
        line_width=60,
        dir=None
    ):
        region = self.reference_region.expand(flank)
        ref_seq = _reference_sequence(reference_fasta, region)
        if ac != 0:
            alt_seq = self._alt_seq(ref_seq, flank)
        
        if ref_contig is None:
            ref_contig = str(region).replace(":", "_").replace("-", "_")
        if alt_contig is None:
            alt_contig = ref_contig + "_alt"
        
        allele_fasta = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".fasta", dir=dir
        )

        # Write out FASTA
        print(">", ref_contig, sep="", file=allele_fasta)
        if ac == 0 or ac == 1:
            for line in textwrap.wrap(ref_seq, width=line_width):
                print(line, file=allele_fasta)
        print(">", alt_contig, sep="", file=allele_fasta)
        if ac == 1 or ac == 2:
            for line in textwrap.wrap(alt_seq, width=line_width):
                print(line, file=allele_fasta)

        return allele_fasta.name, ref_contig, alt_contig

    def as_proto(self):
        sv = npsv2_pb2.StructuralVariant()
        sv.contig = self.contig
        sv.start = self.start
        sv.end = self.end
        sv.svlen = self._record.info["SVLEN"][0]
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

    @property
    def alt_length(self):
        if self._sequence_resolved:
            alt_allele = self._record.alts[0]
            return len(alt_allele)
        else:
            return 1

    def _alt_seq(self, ref_seq, flank):
        if self._sequence_resolved:
            alt_allele = self._record.alts[0]
            return ref_seq[: flank] + alt_allele[self._padding:] + ref_seq[-flank :]
        else:
            return ref_seq[: flank] + ref_seq[-flank :]

    def window_regions(self, window_size: int, flank_windows: int):
        # We anchor windows on the breakpoints
        assert window_size % 2 == 0, "Window size must be evenly split in half"
        breakpoint_flank = window_size // 2
        
        # Create overlapping window in the center of the event if needed
        interior = self.reference_region
        interior_windows = ((interior.length - window_size) // 2) // window_size 
        
        left_region = self.left_flank_region(window_size*flank_windows + breakpoint_flank, breakpoint_flank + interior_windows*window_size)
        right_region = self.right_flank_region(window_size*flank_windows + breakpoint_flank, breakpoint_flank + interior_windows*window_size)
        if left_region.end >= right_region.start:
            # Regions abut or overlap!
            return left_region.window(window_size) + right_region.window(window_size)
        else:
            center_region = interior.center.expand(breakpoint_flank)
            assert center_region.length <= window_size, "Should only be one potentially overlapping 'center' region"
            return left_region.window(window_size) + [center_region] + right_region.window(window_size)