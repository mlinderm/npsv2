import logging, os, tempfile, textwrap
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
        return DeletionVariant(record)

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

    @property
    def reference_range(self):
        return Range(self.contig, self.start + self._padding, self.end)

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
        region = self.reference_range.expand(flank)
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

    def _alt_seq(self, ref_seq, flank):
        alt_allele = self._record.alts[0]
        if self._sequence_resolved:
            return ref_seq[: flank] + alt_allele[self._padding:] + ref_seq[-flank :]
        else:
            return ref_seq[: flank] + ref_seq[-flank :]