import logging, operator
import pysam
from .variant import Variant, _reference_sequence
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


def filter_nonref(vcf_path: str, output_path: str, sample: str, flank=0):
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


def _overlapping_records(vcf_file: pysam.VariantFile, flank=0, drop_multiallelic=True):
    # We assume VCF is in sorted order
    current_range = None
    current_records = []

    for record in vcf_file:
        if drop_multiallelic and len(record.alts) > 1:
            # TODO Add logging message
            continue

        variant = Variant.from_pysam(record)
        variant_range = variant.reference_region.expand(flank)
        if current_range is None:
            current_range = variant_range
            current_records = [record]
        elif current_range.get_overlap(variant_range) > 0:
            current_range = current_range.union(variant_range)
            current_records.append(record)
        else:
            # Next variant doesn't overlap, so yield current records and then reset
            yield current_range, current_records
            current_range = variant_range
            current_records = [record]

    # yield any remaining records
    if current_records:
        yield current_range, current_records


def merge_into_multiallelic(vcf_path: str, output_path: str, reference_fasta: str, flank=0):
    original_records = 0
    merged_records = 0
    
    with pysam.VariantFile(vcf_path) as src_vcf_file:
        # We currently don't take samples into account during the merging
        src_vcf_file.subset_samples([])

        src_header = src_vcf_file.header
        dst_header = src_header.copy()

        # TODO: Overwrite these fields?
        dst_header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">')
        dst_header.add_line(
            '##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference in length between REF and ALT alleles">'
        )

        with pysam.VariantFile(output_path, mode="w", header=dst_header) as dst_vcf_file:

            for merged_range, records in _overlapping_records(src_vcf_file):
                original_records += len(records)
                merged_records += 1

                if len(records) == 1:
                    # No merging needed
                    dst_vcf_file.write(records[0])
                else:
                    # Merge bi-allelic records into multi-allelic records
                    merged_range = merged_range.expand(1, right=0)  # Add at least one padding base

                    alleles = [_reference_sequence(reference_fasta, merged_range)]
                    svlen = []
                    id = []
                    for record in records:
                        variant = Variant.from_pysam(record)

                        variant_range = variant.reference_region
                        left_flank = variant_range.start - merged_range.start
                        right_flank = merged_range.end - variant_range.end
                        assert left_flank >= 1, "There should be at least one padding base"

                        alleles.append(variant._alt_seq(alleles[0], left_flank, allele=1, right_flank=right_flank))
                        svlen.append(len(alleles[-1]) - len(alleles[0]))
                        id.append(record.id)

                    multi_record = dst_header.new_record(
                        contig=merged_range.contig,
                        start=merged_range.start,
                        stop=merged_range.end,
                        alleles=alleles,
                        id=";".join(filter(None, id)),
                    )
                    multi_record_info = multi_record.info
                    multi_record_info.update(
                        {"SVTYPE": "DEL", "SVLEN": svlen,}
                    )
                    dst_vcf_file.write(multi_record)

    logging.info("Merged %d records into %d records", original_records, merged_records)
