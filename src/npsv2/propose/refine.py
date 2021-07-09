import joblib, logging, operator

import pysam
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from . import ORIGINAL_KEY
from ..variant import Variant
from ..range import Range, RangeTree


FEATURES = ["HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST", "DHFFC", "MIN", "DIFF", "BIG CONF", "SMALL CONF"]
KLASS = "MATCHGT"


def _variant_descriptor(record):
    return f"{record.contig}_{record.start}_{record.stop}"


def _record_to_rows(record, orig_min_dist):
    # Convert single VCF record into 1 or more table rows (one row for each sample)
    rows = []
    for i, call in enumerate(record.samples.itervalues()):
        distances = call["DS"]
        # There might be multiple original entries
        originals = np.unique(np.atleast_1d(record.info.get(ORIGINAL_KEY, None)))
        for original in originals:
            rows.append(
                pd.DataFrame(
                    {
                        "ID": record.id,
                        "POS": int(record.pos),
                        "END": record.stop,
                        "ORIGINAL": original,
                        "SV": original or record.id,
                        "SAMPLE": i,
                        "GT": "/".join(map(str, call.allele_indices)),
                        "ORIGINAL_MIN": orig_min_dist[i],
                        "HOMO_REF_DIST": distances[0],
                        "HET_DIST": distances[1],
                        "HOMO_ALT_DIST": distances[2],
                        "DHFFC": call["DHFFC"],
                    },
                    index=[0],
                )
            )
    return pd.concat(rows, ignore_index=True)


def _add_derived_features(table):
    # Compute additional features used to select among possible variants
    table["MIN"] = table[["HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST"]].min(axis=1)
    table["DIFF"] = np.subtract(table.MIN, table.ORIGINAL_MIN)
    table["MIN_RATIO"] = np.divide(table.MIN, table.ORIGINAL_MIN)
    table["BIG CONF"] = table.MIN / table[["HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST"]].median(axis=1)
    table["SMALL CONF"] = table.MIN / table[["HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST"]].max(axis=1)


def _gt_to_alleles(gt):
    # TODO: Handle phased genotypes
    return list(map(int, gt.split("/")))


def _vcf_to_table(src_vcf_file: pysam.VariantFile):
    """Generate Pandas table from pysam.VariantFile

    Args:
        src_vcf_file (pysam.VariantFile): Already opened VCF file

    Returns:
        Tuple of table, original records and alternate records
    """
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

    rows = []
    for id, original_record in original_records.items():
        # Determine minimum "original" distance for each sample
        orig_min_dist = [np.min(call["DS"]) for call in original_record.samples.itervalues()]

        rows.append(_record_to_rows(original_record, orig_min_dist))
        for alt_record in alternate_records.get(id, []):
            rows.append(_record_to_rows(alt_record, orig_min_dist))
    table = pd.concat(rows, ignore_index=True)

    _add_derived_features(table)

    return table, original_records, alternate_records


def refine_vcf(
    cfg,
    vcf_path: str,
    output_path: str,
    classifier_path: str = None,
    progress_bar=False,
    include_orig_ref=True,
    merge_blocks=True,
    include_orig_in_block=False,
    
):
    if cfg.refine.select_algo not in { "original", "ml", "min_distance" }:
        raise ValueError(f"{cfg.refine.select_algo} is not a supported selection algorithm")
    
    # Include reference genotype for original variant or not in minimum calculation
    orig_start_idx = 0 if include_orig_ref else 1

    with pysam.VariantFile(vcf_path) as src_vcf_file:
        # Create header for destination file
        src_header = src_vcf_file.header
        dst_header = src_header.copy()
        dst_header.add_line('##FORMAT=<ID=CL,Number=1,Type=String,Description="Call location used for genotype">')
        dst_header.add_line('##FORMAT=<ID=OGT,Number=1,Type=String,Description="Genotype for the original variant">')
        dst_header.add_line(
            '##FORMAT=<ID=ODS,Number=G,Type=Float,Description="Distance between real and simulated data for the original variant">'
        )
        dst_header.add_line(
            '##FORMAT=<ID=SRC,Number=1,Type=String,Description="Selected other variant in overlapping block">'
        )

        table, original_records, _ = _vcf_to_table(src_vcf_file)

        if cfg.refine.select_algo == "ml":
            # Load the "refine" classifier and predict best proposal
            clf = joblib.load(classifier_path)
            table["PROBTRUE"] = clf.predict_proba(table[FEATURES])[:, 1]
         
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(table)

        # # Determine variant groups
        # variant_ranges = RangeTree()
        # for id, original_record in original_records.items():
        #     total_range = Variant.from_pysam(original_record).reference_region
        #     for alternate_record in alternate_records.get(id, []):
        #         total_range = total_range.union(Variant.from_pysam(alternate_record).reference_region)
        #     variant_ranges.add(total_range, [id])

        # # Optionally merged overlapping variant ranges into single blocks
        # if merge_blocks:
        #     variant_ranges.merge_overlaps(data_reducer=operator.add, data_initializer=[])

        # # Determine the best alternate representation(s) in each group
        # closest_alts = {}
        # for ids in variant_ranges.values():
        #     best_alts = [(float("Inf"), None, None, False)] * num_samples
        #     for id in ids:
        #         possible_alternate_records = alternate_records.get(id, [])
        #         if include_orig_in_block:
        #             possible_alternate_records.append(original_records[id])
        #         for alternate_record in possible_alternate_records:
        #             for i, alternate_call in enumerate(alternate_record.samples.itervalues()):
        #                 best_dist, *_ = best_alts[i]

        #                 alt_dist = alternate_call["DS"]
        #                 min_idx = np.argmin(alt_dist)
        #                 alt_ratio = alternate_call["DHFFC"]
        #                 if min_idx != 0 and alt_dist[min_idx] < best_dist:
        #                     best_alts[i] = (alt_dist[min_idx], id, alternate_record, False)
        #                     if alt_dist[min_idx]/alt_dist[0] < 0.125:
        #                         if min_idx == 1:
        #                             if (alt_dist[min_idx]/alt_dist[2] < 0.125 and 0.4 <= alt_ratio <= 0.6):
        #                                 best_alts[i] = (alt_dist[min_idx], id, alternate_record, True)
        #                         else:
        #                             if (alt_dist[min_idx]/alt_dist[1] < 0.125 and alt_ratio <= 0.2):
        #                                 best_alts[i] = (alt_dist[min_idx], id, alternate_record, True)

        #     for id in ids:
        #         closest_alts[id] = best_alts

        variant_table = table.groupby(["SV", "SAMPLE"])

        with pysam.VariantFile(output_path, mode="w", header=dst_header) as dst_vcf_file:
            for id, record in original_records.items():
                record.translate(dst_header)

                if cfg.refine.select_algo == "original":
                    dst_vcf_file.write(record)  # Just use original SV genotype without trying to refine
                    continue


                for i, call in enumerate(record.samples.itervalues()):
                    possible_calls = variant_table.get_group((id, i)).reset_index(drop=True)

                    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    #     print(possible_calls)

                    if possible_calls.shape[0] == 1:
                        # No alternate record to update with
                        assert pd.isna(possible_calls.loc[possible_calls.index[0], "ORIGINAL"])
                        continue
                    else:
                        if cfg.refine.select_algo == "ml":
                            max_prob_idx = possible_calls.PROBTRUE.idxmax()
                            alt_row = possible_calls.iloc[max_prob_idx, :]
                        else:
                            min_dist_idx = np.unravel_index(np.argmin(possible_calls[["HOMO_REF_DIST","HET_DIST","HOMO_ALT_DIST"]]), (possible_calls.shape[0], 3))
                            alt_row = possible_calls.iloc[min_dist_idx[0], :]

                        if not pd.isna(alt_row.ORIGINAL):
                            call.update(
                                {
                                    "DS": [alt_row.HOMO_REF_DIST, alt_row.HET_DIST, alt_row.HOMO_ALT_DIST],
                                    "DHFFC": alt_row.DHFFC,
                                    "CL": f"{alt_row.POS}_{alt_row.END}",
                                }
                            )
                            call.allele_indices = _gt_to_alleles(alt_row.GT)

                # if id not in closest_alts:
                #     # No alternate records present for this variant, or any blocks this variant overlaps
                #     pass
                # else:
                #     # Identify best alternate representation and genotype for each sample
                #     closest_alt = closest_alts[id]
                #     for i, call in enumerate(record.samples.itervalues()):
                #         alt_dist, alt_id, alt_record, alt_better = closest_alt[i]
                #         if alt_record is None or alt_record is record:
                #             # No alternate record to update with, or we are tying to update with ourselves (same original record)
                #             continue

                #         orig_dist = min(call["DS"][orig_start_idx:])
                #         diff = alt_dist - orig_dist
                #         if diff <= 0.1 and alt_better == True and alt_id == id:
                #             #One of our alternate representations is best, based on coverage criteria
                #             alt_call = alt_record.samples[i]
                #             call.update({
                #                 "SRC": "var",
                #                 "DS": alt_call["DS"],
                #                 "CL": _variant_descriptor(alt_record),
                #                 "OGT": "/".join(map(str, call.allele_indices)),
                #                 "ODS": call["DS"],
                #             })
                #             call.allele_indices = alt_call.allele_indices

                #         elif alt_dist < orig_dist and alt_id == id:
                #             # One of our alternate representations is best, use that alternate genotype
                #             alt_call = alt_record.samples[i]
                #             call.update({
                #                 "SRC": "var",
                #                 "DS": alt_call["DS"],
                #                 "CL": _variant_descriptor(alt_record),
                #                 "OGT": "/".join(map(str, call.allele_indices)),
                #                 "ODS": call["DS"],
                #             })
                #             call.allele_indices = alt_call.allele_indices

                #         elif alt_dist < orig_dist and alt_id != id:
                #             # A different variant's alternate representation is best, set our genotype to 0/0
                #             alt_call = alt_record.samples[i]
                #             call.update({
                #                 "SRC": "blk",
                #                 "CL": _variant_descriptor(alt_record),
                #                 "OGT": "/".join(map(str, call.allele_indices)),
                #                 "ODS": call.pop("DS"),
                #             })
                #             call.allele_indices = [0, 0]

                dst_vcf_file.write(record)


def _supplement_false_training_examples(orig_table, matched_training, method="random"):
    """Supplement training data with additional FALSE entries, if needed, to create balanced classes

    Args:
        orig_table ([type]): [description]
        matched_training ([type]): [description]
        method (str, optional): [description]. Defaults to "random".
    """
    # Supplement training data with additional FALSE entries to create balanced classes
    klass_counts = matched_training.value_counts("MATCHGT")
    delta_true = klass_counts.loc[True] - klass_counts.loc[False]
    assert delta_true >= 0, "Downsampling should only retain FALSE examples associated with a TRUE example"
    if delta_true > 0:
        # Get FALSE examples not already included in the training data
        other_falses = orig_table.loc[orig_table.index.difference(matched_training.index)]
        assert not other_falses[KLASS].any(), "All TRUE examples should be in downsampled training data"
        if method == "random":
            delta_examples = other_falses.sample(n=delta_true)
        elif method == "hard":
            delta_examples = other_falses.nlargest(delta_true, "PROBTRUE")
        else:
            assert False, "Unsupported downsampling method"
        return matched_training.append(delta_examples)
    else:
        return matched_training


def train_model(vcf_path: str, pbsv_file: str, output_path: str):
    """Train model for selecting among proposed SVs

    Args:
        vcf_path (str): Path to VCF file of proposed SVs
        pbsv_file (str): Path to TSV file of SVs that matched long-read derived calls
        output_path (str): Path to save the model
    """
    # Load NSPV2 VCF into Pandas table
    with pysam.VariantFile(vcf_path) as src_vcf_file:
        table, *_ = _vcf_to_table(src_vcf_file)

    # Load table of proposals that matched long-read derived calls and join with proposals
    longread_table = pd.read_csv(
        pbsv_file, sep="\t", na_values=["."], header=0, names=["ID", "ORIGINAL", "POS", "END", KLASS]
    )
    merge_table = pd.merge(table, longread_table, on=["ID", "ORIGINAL", "POS", "END"], how="left")

    avail_data = merge_table.value_counts(KLASS)
    logging.info(
        "Training with %d SVs, %d/%d matched long-read calls with correct/incorrect genotypes",
        merge_table.shape[0],
        avail_data.get(True, 0),
        avail_data.get(False, 0),
    )

    # Any missing MATCHGT values are by definition False since those were not the best proposal
    # TODO: Exclude SVs we couldn't match to any PBSV call?
    merge_table.fillna({"MATCHGT": False}, inplace=True)

    # Perform matched downsampling, i.e. selecting "FALSE" for every SV with "TRUE" or matching proposal
    matched_training = merge_table.groupby("SV").filter(lambda x: x.MATCHGT.any()).groupby(["SV", KLASS]).sample(n=1)
    matched_training = _supplement_false_training_examples(merge_table, matched_training, method="random")

    # First train on randomly-sampled balanced training data
    logging.info("Training initial model on %d observations", matched_training.shape[0])
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    clf.fit(matched_training[FEATURES], matched_training[KLASS])
    assert np.array_equal(clf.classes_, [False, True])

    # Select "hard" falses (since there is only one TRUE per SV, it should always be the max)
    merge_table["PROBTRUE"] = clf.predict_proba(merge_table[FEATURES])[:, 1]
    matched_training = merge_table.loc[
        merge_table.groupby("SV").filter(lambda x: x.MATCHGT.any()).groupby(["SV", KLASS]).PROBTRUE.idxmax()
    ]
    matched_training = _supplement_false_training_examples(merge_table, matched_training, method="hard")

    # Train again using "hard" falses
    logging.info("Training model using 'hard negatives' on %d observations", matched_training.shape[0])
    clf.fit(matched_training[FEATURES], matched_training[KLASS])

    logging.info("Saving model in: %s", output_path)
    joblib.dump(clf, output_path, compress=3)

