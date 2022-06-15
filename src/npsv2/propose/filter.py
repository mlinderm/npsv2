import logging, os
import pysam
from tqdm import tqdm
import dna_jellyfish as jf
from ..variant import Variant, _reference_sequence
from ..utilities.vcf import index_variant_file

from . import ORIGINAL_KEY


def filter_vcf(cfg, vcf_path: str, output: str, progress_bar=False):

    k = cfg.refine.filterk

    # Use Python interface to Jellyfish database
    if not cfg.refine.reference_jf_path or not os.path.exists(cfg.refine.reference_jf_path):
        raise ValueError("Genome jellyfish database not defined")
    ref_query = jf.QueryMerFile(cfg.refine.reference_jf_path)

    if not cfg.refine.reads_jf_path or not os.path.exists(cfg.refine.reads_jf_path):
        raise ValueError("Reads jellyfish database not defined")
    reads_query = jf.QueryMerFile(cfg.refine.reads_jf_path)

    # Stats
    total_records = 0
    original_records = 0
    nonunique_records = 0
    supported_records = 0

    with pysam.VariantFile(vcf_path) as src_vcf_file:
        # Create header for destination file
        src_header = src_vcf_file.header
        dst_header = src_header.copy()

        with pysam.VariantFile(output, mode="w", header=dst_header) as dst_vcf_file:
            for record in tqdm(src_vcf_file, desc="Filtering SVs using k-mers", disable=not progress_bar):
                total_records += 1
                # Write all "original" records out by default, even if they might otherwise be filtered out
                if not record.info.get(ORIGINAL_KEY, False):
                    dst_vcf_file.write(record)
                    original_records += 1
                    continue

                variant = Variant.from_pysam(record)

                region = variant.reference_region.expand(cfg.refine.filterk - 1)
                ref_seq = _reference_sequence(cfg.reference, region)

                unique_kmers = present_kmers = 0
                for allele in variant.alt_allele_indices:
                    alt_seq = variant._alt_seq(ref_seq, cfg.refine.filterk - 1, allele)
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
                            break

                    if present_kmers > 0:
                        break

                assert not (unique_kmers == 0 and present_kmers > 0)
                if unique_kmers == 0:
                    dst_vcf_file.write(record)
                    nonunique_records += 1
                elif present_kmers > 0:
                    dst_vcf_file.write(record)
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
        
        # Write index if file if compressed variant file
        index_variant_file(output)
            
