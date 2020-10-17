from .range import Range

class Variant(object):
    def __init__(self, record):
        self.record = record

    @classmethod
    def from_pysam(cls, record):
        return Variant(record)

    @property
    def contig(self):
        return self.record.contig

    @property
    def start(self):
        return self.record.start

    @property
    def end(self):
        return self.record.stop

    def is_biallelic(self):
        # TODO: Exclude alleles
        return len(self.record.alts) == 1

    @property
    def reference_range(self):
        return Range(self.contig, self.start, self.end)

    def genotype_indices(self, index_or_id):
        call = self.record.samples[index_or_id]
        return call.allele_indices if call else None

