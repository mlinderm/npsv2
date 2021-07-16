import collections, logging, operator, os, sys, tempfile
import pysam
import pysam.bcftools as bcftools
import numpy as np
from scipy.signal import peak_prominences, find_peaks
from tqdm import tqdm

from . import ORIGINAL_KEY
from ..variant import Variant, _reference_sequence
from ..range import Range
from ..utilities.vcf import index_variant_file, bcftools_format


def propose_vcf(cfg, vcf_path: str, output_path: str, repeats_bed_path: str, progress_bar=False):
    """Generate alternate representations for variants in a VCF file.

    Args:
        cfg: Hydra configuration object
        vcf_path (str): Path to input VCF file
        output_path (str): Path to output VCF file
        repeats_bed_path (str): Path to UCSC "simple repeats" BED file
    """
    with tempfile.TemporaryDirectory() as output_dir, pysam.VariantFile(vcf_path) as src_vcf_file:
        # Create header for destination file
        src_header = src_vcf_file.header
        #assert ORIGINAL_KEY not in src_header.info, f"{ORIGINAL_KEY} already presented in VCF INFO field"
        dst_header = src_header.copy()
        dst_header.add_line(
            f'##INFO=<ID={ORIGINAL_KEY},Number=.,Type=String,Description="Proposed alternate representation for these variant IDs">'
        )

        # Setup repeats file
        simple_repeats_bed = pysam.TabixFile(repeats_bed_path)

        unsorted_output_path = os.path.join(output_dir, "proposals.vcf.gz")
        with pysam.VariantFile(unsorted_output_path, mode="w", header=dst_header) as dst_vcf_file:

            proposed_variants = {}
            observed_variants = {}
            for i, record in enumerate(tqdm(src_vcf_file, desc="Generating proposed SV representations", disable=not progress_bar)):
                # Clean up and create unique variant ID to prevent downstream errors
                multiple_ids = record.id.find(";")
                if multiple_ids != -1:
                    record.id = record.id[:multiple_ids]
                if observed_variants.setdefault(record.id, i) != i:
                    record.id += f"_{i}"
                
                record.translate(dst_header)
                dst_vcf_file.write(record)  # Write original record
                variant = Variant.from_pysam(record)

                if not variant.is_deletion:
                    # Only deletions currently supported
                    continue

                length_change = abs(variant.length_change())
                if length_change > cfg.refine.max_propose_length:
                    continue

                # TODO: Require certain overlap?
                repeats = simple_repeats_bed.fetch(region=str(variant.reference_region), parser=pysam.asTuple())

                if not repeats:
                    continue

                for repeat in repeats:
                    consensus_length = int(repeat[3])
                    # Only propose variants for larger VNTRs
                    if consensus_length < cfg.refine.min_consensus_length:
                        continue

                    # Only propose variants if original variant is smaller than repeat region
                    repeat_length = consensus_length * float(repeat[4])
                    if length_change > repeat_length or (repeat_length - length_change) / consensus_length < 1:
                        continue

                    event_repeat_count = round(length_change / consensus_length)
                    if event_repeat_count == 0:
                        continue

                    repeat_start = int(repeat[1]) - cfg.refine.peak_finding_flank
                    repeat_end = int(repeat[2]) + cfg.refine.peak_finding_flank

                    ref_seq = _reference_sequence(cfg.reference, Range(variant.contig, repeat_start, repeat_end))

                    if cfg.refine.all_alignments:
                        peaks = list(range(0, len(ref_seq)))
                    else:
                        consensus_seq = repeat[5]
                        scores = []
                        for i in range(0, len(ref_seq) - len(consensus_seq)):
                            matches = sum(
                                c1 == c2 for c1, c2 in zip(consensus_seq, ref_seq[i : i + len(consensus_seq)])
                            )
                            scores.append(matches)

                        peaks, properties = find_peaks(scores, width=1, distance=consensus_length * 0.8)

                        # Enforce maximum number of potential alternate variants by selecting most prominent
                        peaks = peaks[np.argsort(properties["prominences"])[: cfg.refine.max_proposals]]

                    # Generate alternate records
                    for peak in peaks:
                        # TODO: Realign allele sequence to get better end coordinate?
                        alt_pos = (
                            peak + repeat_start
                        )  # 0-indexed base within event, 1-indexed base immediately before event
                        if cfg.refine.all_alignments:
                            alt_end = peak + repeat_start + event_repeat_count * consensus_length #length_change
                        else:
                            alt_end = peak + repeat_start + event_repeat_count * consensus_length

                        key = Range(variant.contig, alt_pos, alt_end)
                        if key == variant.reference_region:
                            continue  # The same as the original variant

                        if key in proposed_variants:
                            new_record = proposed_variants[key]
                            new_record.info.update(
                                {ORIGINAL_KEY: tuple(np.unique(new_record.info[ORIGINAL_KEY] + (record.id,)))}
                            )
                        else:
                            # TODO: Handle when SVLEN is not a vector
                            proposed_variants[key] = dst_header.new_record(
                                contig=key.contig,
                                start=key.start - 1,  # Padding base
                                stop=key.end,
                                alleles=[ref_seq[peak - 1], "<DEL>"],
                                info={"SVTYPE": "DEL", "SVLEN": [-int(key.length)], ORIGINAL_KEY: [record.id]},
                            )

            for proposed_record in proposed_variants.values():
                dst_vcf_file.write(proposed_record)


        # Sort output file and index (if relevant)
        bcftools.sort("-O", bcftools_format(output_path), "-o", output_path, "-T", output_dir, unsorted_output_path, catch_stdout=False)
        index_variant_file(output_path)
