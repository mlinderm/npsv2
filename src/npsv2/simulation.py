import copy, logging, os, random, re, subprocess, tempfile
import pysam
import numpy as np
from shlex import quote
from .range import Range
from .variant import Variant
from .sample import Sample
from . import _native

CHROM_REGEX_AUTO = r"^(chr)?([1-9][0-9]?)$"
CHROM_REGEX_AUTO_NOY = r"^(chr)?([1-9][0-9]?|[X])$"
CHROM_REGEX_SEX = r"^(chr)?[XY]"

class RandomVariants(object):
    def __init__(self, ref_path: str, exclude_path: str):
        self._ref_reader = pysam.FastaFile(ref_path)  # pylint: disable=no-member
        
        self._linear_genome = [0]
        self._contigs = []
        for contig, length in zip(self._ref_reader.references, self._ref_reader.lengths):
            if re.fullmatch(CHROM_REGEX_AUTO, contig):
                self._contigs.append(contig)
                self._linear_genome.append(self._linear_genome[-1] + length)
            
        self._exclude_reader = pysam.TabixFile(exclude_path) # pylint: disable=no-member

    def _generate_ranges(self, size, n=1, flank=0):
        num_regions = 0
        while num_regions < n:
            linear_start = random.randrange(self._linear_genome[-1])
            linear_bin = np.digitize(linear_start, self._linear_genome)
            start = linear_start - self._linear_genome[linear_bin - 1]

            # Skip regions within flank distance of the contig ends
            if start < flank or (linear_start + size) > (self._linear_genome[linear_bin] - flank):
                continue
            
            region = Range(self._contigs[linear_bin - 1], start, start + size)
            
            # Skip regions that overlap excluded regions
            exclude_iter = self._exclude_reader.fetch(region.contig, region.start, region.end)
            if next(exclude_iter, None):
                continue
            
            yield region
            num_regions += 1

    def _generate_deletions(self, size, n=1, flank=0):
        variant_header = pysam.VariantHeader()
        variant_header.info.add("END", 1, "Integer", "End coordinate of this variant") 
        variant_header.info.add("SVTYPE", 1, "String", "Type of structural variant")
        variant_header.info.add("SVLEN", 1, "Integer", "Difference in length between REF and ALT alleles")
        variant_header.info.add("IMPRECISE", 0, "Flag", "Imprecise structural variation")
        variant_header.info.add("CIEND", 2, "Integer", "Confidence interval around END for imprecise variants")
        variant_header.info.add("CIPOS", 2, "Integer", "Confidence interval around POS for imprecise variants")
   
        for i, contig in enumerate(self._contigs):
            variant_header.contigs.add(contig)
        
        for region in self._generate_ranges(size, n, flank):
            # start is 0-indexed inclusive start, which is equivalent to the 1-indexed padding base, while 
            # end is a 0-indexed exclusive end, which is equivalent to the 1-indexed inclusive end
            padding_start = region.start - 1
            ref_base = self._ref_reader.fetch(region.contig, padding_start, region.start)
            record = variant_header.new_record(contig=region.contig, start=padding_start, stop=region.end, alleles=[ref_base, "<DEL>"], info={"END": region.end, "SVTYPE": "DEL", "SVLEN": -size})
            
            yield Variant.from_pysam(record)

    def generate(self, variant, n=1, flank=0):
        if variant.is_deletion:
            return self._generate_deletions(abs(variant.length_change()), n=n, flank=flank)
        else:
            assert False, "Unsupported variant type"


def _art_read_length(read_length, profile):
    """Make sure read length is compatible ART"""
    if profile in ("HS10", "HS20"):
        return min(read_length, 100)
    elif profile in ("HS25", "HSXn", "HSXt"):
        return min(read_length, 150)
    else:
        return read_length


def simulate_variant_sequencing(fasta_path, allele_count, sample: Sample, reference, shared_reference=None, dir=tempfile.gettempdir()):
    hap_coverage =  sample.mean_coverage / 2
    shared_ref_arg = f"-S {quote(shared_reference)}" if shared_reference else ""
        
    replicate_bam = tempfile.NamedTemporaryFile(delete=False, suffix=".bam", dir=dir)
    replicate_bam.close()

    synth_commandline = f"synthBAM \
        -t {quote(dir)} \
        -R {quote(reference)} \
        {shared_ref_arg} \
        -c {hap_coverage:0.1f} \
        -m {sample.mean_insert_size} \
        -s {sample.std_insert_size} \
        -l {_art_read_length(sample.read_length, sample.sequencer)} \
        -p {sample.sequencer} \
        -i 1 \
        -z {allele_count} \
        {fasta_path} \
        {replicate_bam.name}"

    synth_result = subprocess.run(synth_commandline, shell=True, stderr=subprocess.PIPE)
    if synth_result.returncode != 0 or not os.path.exists(replicate_bam.name):
        raise RuntimeError(f"Synthesis script failed to generate BAM")

    return replicate_bam.name


def filter_reads_gc(stats_path: str, fasta_path: str, in_sam: str, out_fastq: str, max_norm_covg=2.):
    sample = Sample.from_json(stats_path)
        
    gc_covg = np.fromiter((sample.gc_normalized_coverage(gc) for gc in range(0, 101, 1)), dtype=float)
    max_normalized_gc = min(np.max(gc_covg), args.max_norm_covg)
    gc_covg /= max_normalized_gc

    _native.filter_reads_gc(fasta_path, in_sam, out_fastq, gc_covg)


def augment_samples(original_sample: Sample, n, keep_original=True):
    new_samples = [original_sample] if keep_original else []

    # TODO: Apply augmentation approach like snorkel, with multiple augmentor functions. Potential augmentors
    # include coverage, insert-size distribution, GC distribution   
    for _ in range(n - len(new_samples)):
        new_sample = copy.copy(original_sample)

        new_sample.mean_coverage = random.uniform(max(original_sample.mean_coverage - 10, 0), original_sample.mean_coverage + 10)
        #new_sample.mean_insert_size = random.uniform(original_sample.mean_insert_size - 75, original_sample.mean_insert_size + 75)
        #new_sample.std_insert_size = random.uniform(original_sample.std_insert_size - 30, original_sample.std_insert_size + 30)
        
        new_samples.append(new_sample)

    return new_samples