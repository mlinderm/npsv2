import joblib, logging, operator

import pysam
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.special import softmax
from tqdm import tqdm

from . import ORIGINAL_KEY
from ..variant import Variant, allele_indices_to_ac
from ..range import Range, RangeTree
from ..utilities.vcf import index_variant_file
from ..utilities.sequence import as_scalar
from ..images import image_region

FEATURES = ["AC", "HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST", "DHFFC"]
KLASS = "MATCHGT"


def _variant_descriptor(record):
    return f"{record.contig}_{record.start}_{record.stop}"

def _is_biallelic(record: pysam.VariantRecord) -> bool:
    return len(record.alts) == 1


def _record_to_rows(record, orig_min_dist):
    alt_indices = set(range(1,1+len(record.alts)))
    
    # Convert single VCF record into 1 or more table rows (one row for each sample)
    rows = []
    for i, call in enumerate(record.samples.itervalues()):
        distances = call["DS"]
        # There might be multiple original entries
        originals = np.unique(np.atleast_1d(record.info[ORIGINAL_KEY])) if ORIGINAL_KEY in record.info else [None]
        for original in originals:
            rows.append({
                "ID": record.id,
                "POS": int(record.pos),
                "END": record.stop,
                "SVLEN": as_scalar(record.info.get("SVLEN", int(record.pos) - record.stop)),
                "ORIGINAL": original,
                "SV": original or record.id,
                "SAMPLE": i,
                "GT": "/".join(map(str, call.allele_indices)),
                "AC": allele_indices_to_ac(call.allele_indices, alt_indices),
                "ORIGINAL_MIN": orig_min_dist[i],
                "HOMO_REF_DIST": distances[0],
                "HET_DIST": distances[1],
                "HOMO_ALT_DIST": distances[2],
                "DHFFC": call["DHFFC"][0],
            })
    return pd.DataFrame(rows)


def _add_derived_features(table):
    # Compute additional features used to select among possible variants
    table[["HOMO_REF_PROB", "HET_PROB", "HOMO_ALT_PROB"]] = softmax(-table[["HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST"]], axis=1)
    table["MIN"] = table[["HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST"]].min(axis=1)
    table["DIFF"] = np.subtract(table.MIN, table.ORIGINAL_MIN)
    table["MIN_RATIO"] = np.divide(table.MIN, table.ORIGINAL_MIN)
    table["BIG CONF"] = table.MIN / table[["HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST"]].median(axis=1)
    table["SMALL CONF"] = table.MIN / table[["HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST"]].max(axis=1)


def _gt_to_alleles(gt):
    # TODO: Handle phased genotypes
    return list(map(int, gt.split("/")))


def _vcf_to_table(src_vcf_file: pysam.VariantFile, progress_bar=False):
    """Generate Pandas table from pysam.VariantFile

    Args:
        src_vcf_file (pysam.VariantFile): Already opened VCF file

    Returns:
        Tuple of table, original records and alternate records
    """
    original_records = {}
    alternate_records = {}

    for i, record in enumerate(src_vcf_file):
        if ORIGINAL_KEY not in record.info:
            if record.id is None:
                record.id = str(i)
            if record.id in original_records:
                continue
            original_records[record.id] = record
        else:
            assert not record.id
            originals = record.info[ORIGINAL_KEY]
            for original in originals:
                if original in alternate_records:
                    alternate_records[original].append(record)
                else:
                    alternate_records[original] = [record]

    rows = []
    for id, original_record in tqdm(original_records.items(), desc="Reading variants into table", disable=not progress_bar, mininterval=1):
        # We currently skip multi-allelic variants
        if not _is_biallelic(record):
            continue

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
    classifier_path = [],
    progress_bar=False,
):
    if cfg.refine.select_algo not in { "original", "ml", "min_distance", "max_prob", "metric" }:
        raise ValueError(f"{cfg.refine.select_algo} is not a supported selection algorithm")
    
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
            '##FORMAT=<ID=ALTS,Number=1,Type=Integer,Description="Number of alternate variants considered">'
        )
        dst_header.add_line(
            '##FORMAT=<ID=SRC,Number=1,Type=String,Description="Selected other variant in overlapping block">'
        )

        table, original_records, alternate_records = _vcf_to_table(src_vcf_file, progress_bar=progress_bar)

        if cfg.refine.select_algo == "ml":
            # Load the "refine" classifiers and predict best proposal
            for i, path in enumerate(classifier_path):
                clf = joblib.load(path)
                table[f"ML{i}"] = clf.predict_proba(table[clf.feature_names])[:, 1]

        if cfg.refine.group_variants:
            # Determine variant groups
            variant_ranges = RangeTree()
            for id, original_record in original_records.items():
                total_range = Variant.from_pysam(original_record).reference_region.expand(cfg.refine.variant_group_flank)
                for alternate_record in alternate_records.get(id, []):
                    total_range = total_range.union(Variant.from_pysam(alternate_record).reference_region.expand(cfg.refine.variant_group_flank))
                variant_ranges.add(total_range, [id])

            # Merge overlapping blocks
            variant_ranges.merge_overlaps(data_reducer=operator.add, data_initializer=[])

            # Add additional block column to identify and query overlapping groups
            variant_blocks = {}
            for i, ids in enumerate(variant_ranges.values()):
                for id in ids:
                    variant_blocks[id] = i
            table = table.merge(pd.DataFrame({ "SV": variant_blocks.keys(), "BLOCK": variant_blocks.values() }), how="left", on="SV")
            
            variant_table = table.groupby(["BLOCK", "SAMPLE"])
            def _get_possible_calls(id, sample_index):
                return variant_table.get_group((variant_blocks[id], sample_index)).reset_index(drop=True)
        else:
            variant_table = table.groupby(["SV", "SAMPLE"])
            def _get_possible_calls(id, sample_index):
                return variant_table.get_group((id, sample_index)).reset_index(drop=True)


        with pysam.VariantFile(output_path, mode="w", header=dst_header) as dst_vcf_file:
            # Since Python dictionaries iterate in insertion order, if original dictionary was sorted, so is the output
            for id, record in tqdm(original_records.items(), desc="Refining SV description", disable=not progress_bar, mininterval=1):
                record.translate(dst_header)

                if cfg.refine.select_algo == "original":
                    dst_vcf_file.write(record)  # Just use original SV genotype without trying to refine
                    continue

                if not _is_biallelic(record):
                    dst_vcf_file.write(record)  # We currently don't refine multi-allelic variants
                    continue

                #original_region = image_region(cfg, Variant.from_pysam(record).reference_region)
                original_region = Variant.from_pysam(record).reference_region

                for i, call in enumerate(record.samples.itervalues()):
                    possible_calls = _get_possible_calls(id, i)
                    if possible_calls.shape[0] == 1:
                        # No alternate record to update with
                        assert pd.isna(possible_calls.loc[possible_calls.index[0], "ORIGINAL"])
                        continue
                    elif not cfg.refine_nonref_orig and np.argmin(call["DS"]) > 0:
                        # Original call is non-reference, respect that genotype (instead of trying to refine)
                        continue
                    else:
                        # Record the number the alternates considered, even we don't pick a different SV
                        call["ALTS"] = possible_calls.shape[0] - 1
                        
                        # Smooth distances (and recalculate Softmax probabilities)
                        if cfg.refine.smooth_distances:
                            # Smooth by ID and SVLEN, so only the same putative SV are smoothed when the are multiple proposed lengths for the same
                            # original SV
                            smoothed_calls = possible_calls.drop(["SV", "SVLEN", "HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST"], axis=1).join(
                                possible_calls
                                    .sort_values(by=["POS"])
                                    .groupby(["SV", "SVLEN"])[["HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST"]]
                                    .apply(lambda x: x.ewm(span=max(x.shape[0] // 10, 10), axis=0).mean())
                                    .reset_index(level=[0, 1])
                            )
                            smoothed_calls[["HOMO_REF_PROB", "HET_PROB", "HOMO_ALT_PROB"]] = softmax(-smoothed_calls[["HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST"]], axis=1)
                            possible_calls = smoothed_calls
                            # TODO: Update GT based on smoothed distances?

                        if cfg.refine.select_algo == "ml":
                            # Select the row with the largest probability of being a true update if
                            # 1) the SV has a non-reference genotype, and 
                            # 2) ...
                            orig = possible_calls[possible_calls.ID == id].squeeze()
                            assert len(orig.shape) == 1
                            
                            nonref_possible_calls = possible_calls[possible_calls.GT != "0/0"]
                            if nonref_possible_calls.shape[0] == 0:
                                continue

                            # Find the most likely SV for each model
                            alt_probs = nonref_possible_calls[[f"ML{i}" for i in range(len(classifier_path))]]
                            nonref_possible_calls = nonref_possible_calls.loc[alt_probs.idxmax(axis=0).unique()]
                            
                            # Pick the minimum distance among the most likely SVs
                            min_dist_idx = nonref_possible_calls.MIN.idxmin()
                            alt_row = nonref_possible_calls.loc[min_dist_idx]
                            
                             # Only update if probability for alternate SV is better than the original for the model we used
                            alt_row_model = alt_probs.loc[min_dist_idx].idxmax()
                            if alt_row[alt_row_model] < orig[alt_row_model]:
                                continue
                                                      
                        elif cfg.refine.select_algo == "min_distance":
                            # Select the row with smallest non-reference distance, if that distance is
                            # 1) less than the hom. ref. distance for that SV, and
                            # 2) less than (or equal?) the minimum distance for the original SV. (or equal used for updating when using all originals...)
                            alt_dists = possible_calls[["HET_DIST","HOMO_ALT_DIST"]].min(axis=1)
                            nonref_possible_calls = possible_calls.loc[(alt_dists <= possible_calls.HOMO_REF_DIST),:]
                            if nonref_possible_calls.shape[0] == 0:
                                continue
                            
                            min_dist_idx = np.unravel_index(np.argmin(nonref_possible_calls[["HET_DIST","HOMO_ALT_DIST"]]), (nonref_possible_calls.shape[0], 2))
                            alt_row = nonref_possible_calls.iloc[min_dist_idx[0], :]
                            
                            # Does this alternate record overlap the original record? If so include hom. ref. distance for original record,
                            # if not, optionally don't since large offsets (which don't overlap the SV) should look like hom. ref.
                            #alt_region = image_region(cfg, Range(original_region.contig, alt_row.POS, alt_row.END))
                            alt_region = Range(original_region.contig, alt_row.POS, alt_row.END)
                            if original_region.get_overlap(alt_region) >= cfg.refine.include_orig_ref_min_overlap or cfg.refine.include_orig_ref:
                                orig_min = np.min(call["DS"])
                            else:
                                orig_min = np.min(call["DS"][1:])

                            if alt_row.MIN > orig_min:
                                continue

                        elif cfg.refine.select_algo == "max_prob":
                            # Select the row with largest non-reference probability, if that probability is
                            # 1) greater than the hom. ref. prob for that SV, and
                            # 2) greater than (or equal?) the maximum probability for the original SV. (or equal used for updating when using all originals...)
                            alt_probs = possible_calls[["HET_PROB","HOMO_ALT_PROB"]].max(axis=1)
                            nonref_possible_calls = possible_calls.loc[(alt_probs >= possible_calls.HOMO_REF_PROB),:]
                            if nonref_possible_calls.shape[0] == 0:
                                continue

                            max_prob_idx = np.unravel_index(np.argmax(nonref_possible_calls[["HET_PROB","HOMO_ALT_PROB"]]), (nonref_possible_calls.shape[0], 2))
                            alt_row = nonref_possible_calls.iloc[max_prob_idx[0], :]
                            
                            # Does this alternate record overlap the original record? If so include hom. ref. probability for original record,
                            # if not, optionally don't since large offsets (which don't overlap the SV) should look like hom. ref.
                            #alt_region = image_region(cfg, Range(original_region.contig, alt_row.POS, alt_row.END))
                            alt_region = Range(original_region.contig, alt_row.POS, alt_row.END)
                            orig_probs = softmax(-np.array(call["DS"]))
                            if original_region.get_overlap(alt_region) >= cfg.refine.include_orig_ref_min_overlap or cfg.refine.include_orig_ref:
                                orig_prob = np.max(orig_probs)
                            else:
                                orig_prob = np.min(orig_probs[1:])

                            if np.max(alt_row[["HOMO_REF_PROB","HET_PROB","HOMO_ALT_PROB"]]) < orig_prob:
                                continue         

                        elif cfg.refine.select_algo == "metric":
                            # Select the row with smallest non-reference distance, if that distance is
                            # 1) less than the hom. ref. distance for that SV, and
                            # 2) less than (or equal?) the minimum distance for the original SV. (or equal used for updating when using all originals...)
                            alt_dists = possible_calls[["HET_DIST","HOMO_ALT_DIST"]].min(axis=1)
                            nonref_possible_calls = possible_calls.loc[(alt_dists <= possible_calls.HOMO_REF_DIST),:]
                            if nonref_possible_calls.shape[0] == 0:
                                continue
                            
                            metric = np.sqrt(np.square(nonref_possible_calls[["HET_DIST","HOMO_ALT_DIST"]].min(axis=1)) + np.square(1-nonref_possible_calls.HOMO_REF_DIST))
                            alt_row = nonref_possible_calls.iloc[np.argmin(metric), :]
                            
                            # Does this alternate record overlap the original record? If so include hom. ref. distance for original record,
                            # if not, optionally don't since large offsets (which don't overlap the SV) should look like hom. ref.
                            alt_region = Range(original_region.contig, alt_row.POS, alt_row.END)
                            if original_region.get_overlap(alt_region) >= cfg.refine.include_orig_ref_min_overlap or cfg.refine.include_orig_ref:
                                orig_min = np.min(call["DS"])
                            else:
                                orig_min = np.min(call["DS"][1:])

                            if alt_row.MIN > orig_min:
                                continue

                        # Only update if there is a best row and it is not the same as the original record and one of the following is true:
                        # 1) The original is hom. ref., or we will update non-reference calls
                        # 2) Another variant in the block is best
                        if alt_row is not None and alt_row.ID != id and (cfg.refine.refine_nonref_orig or np.argmin(call["DS"]) == 0 or alt_row.SV != id):
                            call.update(
                                {
                                    "DS": [alt_row.HOMO_REF_DIST, alt_row.HET_DIST, alt_row.HOMO_ALT_DIST],
                                    "DHFFC": alt_row.DHFFC,
                                    "OGT": "/".join(map(str, call.allele_indices)),
                                    "ODS": call["DS"],
                                    "ALTS": possible_calls.shape[0] - 1,
                                    "CL": f"{alt_row.POS}_{alt_row.END}",
                                }
                            )
                            # If another SV in block is the best, set this GT to homozygous reference (TODO: Handle other ploidy)
                            call.allele_indices = _gt_to_alleles(alt_row.GT) if alt_row.SV == id else (0,0)

                dst_vcf_file.write(record)

        # Write index if file if compressed variant file
        index_variant_file(output_path)


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
        #assert not other_falses[KLASS].any(), "All TRUE examples should be in downsampled training data"
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
    clf.feature_names = FEATURES

    logging.info("Saving model in: %s", output_path)
    joblib.dump(clf, output_path, compress=3)

