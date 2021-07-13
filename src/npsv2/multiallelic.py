import operator
import pysam
from .variant import Variant
from .range import Range, RangeTree

def _non_ref_gt(allele_indices):
    for allele_index in allele_indices:
        if allele_index > 0:
            return True
    return False

def _write_non_ref_records(vcf_file, records, flank):
    # Any non-reference entries?
    non_ref = [record for record in records if _non_ref_gt(record.samples[0].allele_indices)]
    if len(non_ref) == 0:
        # Write out all the non-reference entries in the overlapping region
        for record in records:
           vcf_file.write(record)
    elif len(non_ref) == 1:
        # Write out the single non-ref entry
        for record in non_ref:
            vcf_file.write(record)
    else:
        # Write out non-reference variants that no longer overlap
        nonref_ranges = RangeTree()
        for record in non_ref:
            variant_range = Variant.from_pysam(record).reference_region.expand(flank)
            nonref_ranges.add(variant_range, [record])  
        nonref_ranges.merge_overlaps(data_reducer=operator.add, data_initializer=[])
        for grouped_records in nonref_ranges.values():
            # Only write out fully isolated records
            if len(grouped_records) == 1:
                vcf_file.write(grouped_records[0])

def filter_nonref(vcf_path: str, output_path: str, sample: str, flank = 0):
    with pysam.VariantFile(vcf_path) as src_vcf_file:
        src_vcf_file.subset_samples([sample])

        src_header = src_vcf_file.header
        dst_header = src_header.copy()

        with pysam.VariantFile(output_path, mode="w", header=dst_header) as dst_vcf_file:

            # We assume VCF is in sorted order
            current_range = None
            current_records = []
            for record in src_vcf_file:
                alleles = record.samples[0].allele_indices
                if None in alleles:
                    continue  # Skip variants with incomplete genotypes
                
                variant_range = Variant.from_pysam(record).reference_region.expand(flank)
                if current_range is None:
                    current_range = variant_range
                    current_records = [record]
                elif current_range.get_overlap(variant_range) > 0:
                    current_range = current_range.union(variant_range)
                    current_records.append(record)
                else:
                    # Next variant doesn't overlap. Write current records and then reset.
                    _write_non_ref_records(dst_vcf_file, current_records, flank)
                    
                    current_range = variant_range
                    current_records = [record]

            # Write any remaining records
            _write_non_ref_records(dst_vcf_file, current_records, flank)
            
