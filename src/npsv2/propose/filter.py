import logging, os, tempfile, typing
from dataclasses import dataclass
from functools import partial
import pysam
import pysam.bcftools as bcftools
import ray
from tqdm import tqdm
import dna_jellyfish as jf
from ..variant import Variant, _reference_sequence
from ..utilities.vcf import index_variant_file, bcftools_format
from ..range import Range

from . import ORIGINAL_KEY


@dataclass
class KmerStats:
    unique_kmers: int = 0
    present_kmers: int = 0
    original: bool = False


def retain_proposal(
    reference: str, ref_query: jf.QueryMerFile, reads_query: jf.QueryMerFile, k: int, variant: Variant
) -> typing.Tuple[bool, KmerStats]:
    region = variant.reference_region.expand(k - 1)
    ref_seq = _reference_sequence(reference, region)

    unique_kmers = present_kmers = 0
    for allele in variant.alt_allele_indices:
        alt_seq = variant._alt_seq(ref_seq, k - 1, allele)
        for i in range(len(alt_seq) - k + 1):
            mer = jf.MerDNA(alt_seq[i : i + k])

            # The reference genome is only the forward strand and so mers should not be "canonicalized"
            if ref_query[mer] > 0:
                continue
            unique_kmers += 1

            # Only query reads for unique kmers. If any supporting data, is found, immediately write variant.
            # Since we are starting with alignment files, all reads should be on the forward strand and so mers
            # should not be "canonicalized".
            if reads_query[mer] > 0:
                present_kmers += 1
                return True, KmerStats(unique_kmers, present_kmers)

    # Also retain proposal with no unique k-mers
    return unique_kmers == 0, KmerStats(unique_kmers, present_kmers)


def filter_vcf(cfg, vcf_path: str, output: str, progress_bar=False, region: str = None):
    if not cfg.refine.reference_jf_path or not os.path.exists(cfg.refine.reference_jf_path):
        raise ValueError("Genome jellyfish database not defined")
    if not cfg.refine.reads_jf_path or not os.path.exists(cfg.refine.reads_jf_path):
        raise ValueError("Reads jellyfish database not defined")

    if region:
        logging.info("Filtering variants in region %s", region)
        region_args = Range.parse_literal(region).pysam_fetch
    else:
        region_args = {}

    # Use multiple iterators to avoid serialized PySAM objects
    def _vcf_shard(num_shards: int, index: int, yield_record: bool = False):
        if not yield_record:
            # Use Python interface to Jellyfish database
            ref_query = jf.QueryMerFile(cfg.refine.reference_jf_path)
            reads_query = jf.QueryMerFile(cfg.refine.reads_jf_path)
        with pysam.VariantFile(vcf_path) as vcf_file:
            for i, record in enumerate(vcf_file.fetch(**region_args)):
                if i % num_shards != index:
                    continue
                if yield_record:
                    yield (i, record)
                else:
                    # Keep all "original" records, regardless of support
                    if not record.info.get(ORIGINAL_KEY, False):
                        yield (i, True, KmerStats(0, 0, True))
                    else:
                        variant = Variant.from_pysam(record)
                        keep_variant, stats = retain_proposal(
                            cfg.reference, ref_query, reads_query, cfg.refine.filterk, variant
                        )
                        yield (i, keep_variant, stats)

    # Create header for destination file
    with pysam.VariantFile(vcf_path) as src_vcf_file:
        src_header = src_vcf_file.header
        dst_header = src_header.copy()

    # Stats
    total_records = 0
    original_records = 0
    nonunique_records = 0
    supported_records = 0

    with tempfile.TemporaryDirectory() as output_dir:
        # We currently just use ray for the CPU-side work. We use a private temporary directory
        # to avoid conflicts between clusters running on the same node.
        logging.info("Initializing ray with %d threads", cfg.threads)
        # TODO: Seem to run into memory issues starting multiple clusters. Set memory based on allocation?
        ray.init(
            num_cpus=cfg.threads, num_gpus=0, _temp_dir=output_dir, ignore_reinit_error=True, include_dashboard=False
        )

        unsorted_output_path = os.path.join(output_dir, "genotypes.vcf.gz")
        with pysam.VariantFile(unsorted_output_path, mode="wz", header=dst_header) as dst_vcf_file:

            num_shards = cfg.threads
            retain_it = ray.util.iter.from_iterators([partial(_vcf_shard, num_shards, i) for i in range(num_shards)])
            record_its = [_vcf_shard(num_shards, i, yield_record=True) for i in range(num_shards)]

            for index, keep_variant, stats in tqdm(
                retain_it.gather_async(), desc="Filtering SVs using k-mers", disable=not progress_bar
            ):
                # Obtain the corresponding VCF record
                record_index, record = next(record_its[index % num_shards])
                assert record_index == index, "Mismatch VCF variant and result from threading"

                total_records += 1
                if keep_variant:
                    dst_vcf_file.write(record)
                    if stats.original:
                        original_records += 1
                    elif stats.unique_kmers == 0:
                        nonunique_records += 1
                    elif stats.present_kmers > 0:
                        supported_records += 1

            retained_records = original_records + nonunique_records + supported_records
            logging.info(
                "Retained %d of %d (%f) records (%d original, %d non-unique, %d supported)",
                retained_records,
                total_records,
                retained_records / total_records,
                original_records,
                nonunique_records,
                supported_records,
            )

        # Sort output file and index (if relevant)
        bcftools.sort(
            "-O", bcftools_format(output), "-o", output, "-T", output_dir, unsorted_output_path, catch_stdout=False
        )
        index_variant_file(output)
