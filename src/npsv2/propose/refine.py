import operator
import pysam
import numpy as np
from tqdm import tqdm
from ..variant import Variant
from ..range import Range, RangeTree

ORIGINAL_KEY = "ORIGINAL"

def _variant_descriptor(record):
    return f"{record.contig}_{record.start}_{record.stop}"



def refine_vcf(cfg, vcf_path: str, output_path: str, progress_bar=False, include_orig_ref=True, merge_blocks=True, include_orig_in_block=False):
    # Include reference genotype for original variant or not in minimum calculation
    orig_start_idx = 0 if include_orig_ref else 1
    
    with pysam.VariantFile(vcf_path) as src_vcf_file:
        # Create header for destination file
        src_header = src_vcf_file.header
        dst_header = src_header.copy()
        dst_header.add_line('##FORMAT=<ID=CL,Number=1,Type=String,Description="Call location used for genotype">')
        dst_header.add_line('##FORMAT=<ID=OGT,Number=1,Type=String,Description="Genotype for the original variant">')
        dst_header.add_line('##FORMAT=<ID=ODS,Number=G,Type=Float,Description="Distance between real and simulated data for the original variant">')
        dst_header.add_line('##FORMAT=<ID=SRC,Number=1,Type=String,Description="Selected other variant in overlapping block">')


        num_samples = len(dst_header.samples)

        original_records = {}
        alternate_records = {}

        for record in src_vcf_file:
            if ORIGINAL_KEY not in record.info:
                assert record.id and record.id not in original_records, "Duplicate original variants"
                original_records[record.id] = record
            else:
                originals = record.info[ORIGINAL_KEY]
                for original in originals:
                    if original in alternate_records:
                        alternate_records[original].append(record)
                    else:
                        alternate_records[original] = [record]

        
        # Determine variant groups 
        variant_ranges = RangeTree()
        for id, original_record in original_records.items():
            total_range = Variant.from_pysam(original_record).reference_region
            for alternate_record in alternate_records.get(id, []):
                total_range = total_range.union(Variant.from_pysam(alternate_record).reference_region)
            variant_ranges.add(total_range, [id])

        # Optionally merged overlapping variant ranges into single blocks
        if merge_blocks:    
            variant_ranges.merge_overlaps(data_reducer=operator.add, data_initializer=[])

        # Determine the best alternate representation(s) in each group
        closest_alts = {}
        for ids in variant_ranges.values():
            best_alts = [(float("Inf"), None, None)] * num_samples
            for id in ids:
                possible_alternate_records = alternate_records.get(id, [])
                if include_orig_in_block:
                    possible_alternate_records.append(original_records[id])
                for alternate_record in possible_alternate_records:
                    for i, alternate_call in enumerate(alternate_record.samples.itervalues()):
                        best_dist, *_ = best_alts[i]
                        
                        alt_dist = alternate_call["DS"]
                        min_idx = np.argmin(alt_dist)
                        if min_idx != 0 and alt_dist[min_idx] < best_dist:
                            best_alts[i] = (alt_dist[min_idx], id, alternate_record)

            for id in ids:
                closest_alts[id] = best_alts

        with pysam.VariantFile(output_path, mode="w", header=dst_header) as dst_vcf_file:
            for id, record in original_records.items():
                record.translate(dst_header)
                
                if id not in closest_alts:
                    # No alternate records present for this variant, or any blocks this variant overlaps
                    pass
                else:
                    # Identify best alternate representation and genotype for each sample
                    closest_alt = closest_alts[id]
                    for i, call in enumerate(record.samples.itervalues()):
                        alt_dist, alt_id, alt_record = closest_alt[i]
                        if alt_record is None or alt_record is record:
                            # No alternate record to update with, or we are tying to update with ourselves (same original record)
                            continue

                        orig_dist = min(call["DS"][orig_start_idx:])
                        if alt_dist < orig_dist and alt_id == id:
                            # One of our alternate representations is best, use that alternate genotype
                            alt_call = alt_record.samples[i]    
                            call.update({
                                "SRC": "var",
                                "DS": alt_call["DS"],
                                "CL": _variant_descriptor(alt_record),
                                "OGT": "/".join(map(str, call.allele_indices)),
                                "ODS": call["DS"],
                            })
                            call.allele_indices = alt_call.allele_indices
                        elif alt_dist < orig_dist and alt_id != id:
                            # A different variant's alternate representation is best, set our genotype to 0/0
                            alt_call = alt_record.samples[i]
                            call.update({
                                "SRC": "blk",
                                "CL": _variant_descriptor(alt_record),
                                "OGT": "/".join(map(str, call.allele_indices)),
                                "ODS": call.pop("DS"),
                            })
                            call.allele_indices = [0, 0]
                    
                dst_vcf_file.write(record)



