import joblib, logging, operator

import pysam
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from tqdm import tqdm
import os

from . import ORIGINAL_KEY
from ..variant import Variant
from ..range import Range, RangeTree
from ..utilities.vcf import index_variant_file
from scipy.special import softmax

FEATURES = ["HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST", "DHFFC", "AC"]

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
                        "SVLEN": record.info["SVLEN"],
                        "ORIGINAL": original,
                        "SV": original or record.id,
                        "SAMPLE": i,
                        "GT": "/".join(map(str, call.allele_indices)),
                        "ORIGINAL_MIN": orig_min_dist[i],
                        "HOMO_REF_DIST": distances[0],
                        "HET_DIST": distances[1],
                        "HOMO_ALT_DIST": distances[2],
                        "DHFFC": call["DHFFC"],
                        "AC": _count_alleles(call.allele_indices),
                    },
                    index=[0],
                )
            )
    return pd.concat(rows, ignore_index=True)

def _count_alleles(allele_indices):
    ac = 0
    for allele in allele_indices:
        if allele > 0:
            ac += 1
    return ac


def _add_derived_features(table):
    # Compute additional features used to select among possible variants
    table["MIN"] = table[["HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST"]].min(axis=1)
    table["DIFF"] = np.subtract(table.MIN, table.ORIGINAL_MIN)
    table["MIN_RATIO"] = np.divide(table.MIN, table.ORIGINAL_MIN)
    table["BIG CONF"] = table.MIN / table[["HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST"]].median(axis=1)
    table["SMALL CONF"] = table.MIN / table[["HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST"]].max(axis=1)
    
    # Compute softmax and GQ
    softmax_array = softmax(-table[["HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST"]], axis=1)
    table["SOFTMAX_HOMO_REF"] = softmax_array["HOMO_REF_DIST"]
    table["SOFTMAX_HET"] = softmax_array["HET_DIST"]
    table["SOFTMAX_HOMO_ALT"] = softmax_array["HOMO_ALT_DIST"]
    table["GQ"] = table[["SOFTMAX_HOMO_REF", "SOFTMAX_HET", "SOFTMAX_HOMO_ALT"]].max(axis=1) - table[["SOFTMAX_HOMO_REF", "SOFTMAX_HET", "SOFTMAX_HOMO_ALT"]].median(axis=1)


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

    for record in tqdm(src_vcf_file, desc="Reading variants into table", disable=not progress_bar, mininterval=1):
        if ORIGINAL_KEY not in record.info:
            if not record.id or record.id in original_records:
                continue
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
    classifier_path: str, 
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
        dst_header.add_line('##FORMAT=<ID=PCL,Number=1,Type=Float,Description="Probability that selected call is true.">')
        dst_header.add_line('##FORMAT=<ID=OGT,Number=1,Type=String,Description="Genotype for the original variant">')
        dst_header.add_line(
            '##FORMAT=<ID=ODS,Number=G,Type=Float,Description="Distance between real and simulated data for the original variant">'
        )
        dst_header.add_line(
            '##FORMAT=<ID=SRC,Number=1,Type=String,Description="Selected other variant in overlapping block">'
        )
        dst_header.add_line(
            '##FORMAT=<ID=PALT,Number=.,Type=Float,Description="Probabilities of other proposals">'
        )

        table, original_records, _ = _vcf_to_table(src_vcf_file, progress_bar=progress_bar)

        if cfg.refine.select_algo == "ml":
            for i, path in enumerate(classifier_path):
                clf = joblib.load(path)
                table[f"PROBTRUE{i}"] = clf.predict_proba(table[clf.feature_names])[:, 1]
            # base_dir = os.path.join(classifier_path, "base_models")
            # for i, filename in enumerate(os.listdir(base_dir)):
            #     path = os.path.join(base_dir, filename)
            #     clf = joblib.load(path)
            #     table[f"PROBTRUE{i}"] = clf.predict_proba(table[clf.feature_names])[:, 1]
            
            # blender = joblib.load(os.path.join(classifier_path, "blender.joblib"))
            # table["BLEND_PROB"] = blender.predict_proba(table[blender.feature_names])[:, 1]

        variant_table = table.groupby(["SV", "SAMPLE"])

        with pysam.VariantFile(output_path, mode="w", header=dst_header) as dst_vcf_file:
            # Since Python dictionaries iterate in insertion order, if original dictionary was sorted, so is the output
            for id, record in tqdm(original_records.items(), desc="Refining SV description", disable=not progress_bar, mininterval=1):
                record.translate(dst_header)

                if cfg.refine.select_algo == "original":
                    dst_vcf_file.write(record)  # Just use original SV genotype without trying to refine
                    continue

                for i, call in enumerate(record.samples.itervalues()):
                    possible_calls = variant_table.get_group((id, i)).reset_index(drop=True).sort_values(by=["POS"])
                    orig = possible_calls[pd.isna(possible_calls.ID) == False].iloc[0]

                    if possible_calls.shape[0] == 1:
                        # No alternate record to update with
                        assert pd.isna(possible_calls.loc[possible_calls.index[0], "ORIGINAL"])
                        # rows1.append(possible_calls.iloc[0].to_frame())
                        continue
                    else:
                        # if cfg.refine.select_algo == "ml":
                        #     best_calls = possible_calls[possible_calls.GT != "0/0"]
                        #     if best_calls.shape[0] > 0:
                        #         max_prob_idx = best_calls.BLEND_PROB.idxmax()
                        #         alt_row = best_calls.loc[max_prob_idx, :]
                        #         if alt_row.BLEND_PROB < orig.BLEND_PROB:
                        #             alt_row = orig
                        #     else:
                        #         alt_row = orig                        
                        if cfg.refine.select_algo == "ml":
                            best_calls = possible_calls[possible_calls.GT != "0/0"]
                            if best_calls.shape[0] > 0:
                                max_prob_idx = best_calls.PROBTRUE0.idxmax()
                                alt_row = best_calls.loc[max_prob_idx, :]
                                alt_probtrue = alt_row.PROBTRUE0
                                orig_probtrue = orig.PROBTRUE0
                                for i in range(len(classifier_path)):
                                    max_prob_idx2 = best_calls[f"PROBTRUE{i}"].idxmax()
                                    if max_prob_idx != max_prob_idx2 and best_calls.loc[max_prob_idx2, :].MIN < alt_row.MIN:
                                        max_prob_idx = max_prob_idx2
                                        alt_row = best_calls.loc[max_prob_idx2, :]
                                        alt_probtrue = alt_row[f"PROBTRUE{i}"]
                                        orig_probtrue = orig[f"PROBTRUE{i}"]
                                if alt_probtrue < orig_probtrue:
                                    alt_row = orig
                            else:
                                alt_row = orig
                        else:
                            min_dist_idx = np.unravel_index(np.argmin(possible_calls[["HOMO_REF_DIST","HET_DIST","HOMO_ALT_DIST"]]), (possible_calls.shape[0], 3))
                            alt_row = possible_calls.iloc[min_dist_idx[0], :] if min_dist_idx[1] > 0 else None

                        if alt_row is not None and not pd.isna(alt_row.ORIGINAL):
                            #TODO: print out probtrue corresponding to chosen model for given SV (currently always prints probtrue corresponding to first model)
                            call.update(
                                {
                                    "DS": [alt_row.HOMO_REF_DIST, alt_row.HET_DIST, alt_row.HOMO_ALT_DIST],
                                    "DHFFC": alt_row.DHFFC,
                                    "PCL": alt_row.PROBTRUE0, 
                                    "CL": f"{alt_row.POS}__{alt_row.END}",
                                    "PALT": possible_calls.PROBTRUE0.tolist(),
                                }
                            )
                            call.allele_indices = _gt_to_alleles(alt_row.GT)

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

    #create model to select "hard" falses
    clf = KNeighborsClassifier()
    clf.fit(matched_training[FEATURES], matched_training[KLASS])
    clf.feature_names = FEATURES
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

def ensemble_train(vcf_path: str, pbsv_file: str, output_path: str):
    """Train ensemble of models for selecting among proposed SVs

    Args:
        vcf_path (str): Path to VCF file of proposed SVs
        pbsv_file (str): Path to TSV file of SVs that matched long-read derived calls
        output_path (str): Directory to save the models in. Base models saved in subdirectory called base_models, blender saved in directory listed as output_path.
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
    clfs = []
    clf0 = KNeighborsClassifier()
    clf1 = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    clf2 = LogisticRegression(random_state=0)
    clfs.append(clf0)
    clfs.append(clf1)
    clfs.append(clf2)

    #create initial model to select "hard" falses
    clf0.fit(matched_training[FEATURES], matched_training[KLASS])
    clf0.feature_names = FEATURES
    assert np.array_equal(clf0.classes_, [False, True])

    # Select "hard" falses (since there is only one TRUE per SV, it should always be the max)
    merge_table["PROBTRUE0"] = clf0.predict_proba(merge_table[FEATURES])[:, 1]
    matched_training = merge_table.loc[
        merge_table.groupby("SV").filter(lambda x: x.MATCHGT.any()).groupby(["SV", KLASS]).PROBTRUE0.idxmax()
    ]
    matched_training = _supplement_false_training_examples(merge_table, matched_training, method="hard")

    #split data in attempt to avoid overfitting - model still overfit
    x_train, x_test, y_train, y_test = train_test_split(matched_training[FEATURES + ["PROBTRUE0"]], matched_training[KLASS], test_size=0.15, random_state=42)

    BLENDER_FEATURES = ["PROBTRUE0"]

    # Train again using "hard" falses
    logging.info("Training model using 'hard negatives' on %d observations", matched_training.shape[0])

    #Create additional models
    for i, clf in enumerate(clfs[1:]):
        clf.fit(x_train[FEATURES], y_train)
        clf.feature_names = FEATURES
        matched_training[f"PROBTRUE{i+1}"] = clf.predict_proba(matched_training[FEATURES])[:, 1]
        x_train[f"PROBTRUE{i+1}"] = clf.predict_proba(x_train[FEATURES])[:, 1]
        x_test[f"PROBTRUE{i+1}"] = clf.predict_proba(x_test[FEATURES])[:, 1]
        assert np.array_equal(clf.classes_, [False, True])
        BLENDER_FEATURES.append(f"PROBTRUE{i+1}")

    #create blender
    blender = LogisticRegression(random_state=0)
    blender.feature_names = BLENDER_FEATURES
    blender.fit(matched_training[BLENDER_FEATURES], matched_training[KLASS])
    blender.fit(x_test[BLENDER_FEATURES], y_test)
    df = pd.DataFrame()
    df['MATCHGT'] = y_train
    df['PREDICTIONS'] = blender.predict(x_train[BLENDER_FEATURES])
    correct_predictions = df[df['PREDICTIONS'] == df['MATCHGT']]
    print(correct_predictions.shape[0]/df.shape[0])
    # y_pred = cross_val_predict(blender, x_train[BLENDER_FEATURES], y_train, cv = 5)
    # y_pred_score = cross_val_score(blender, x_train[BLENDER_FEATURES], y_train, cv = 5)
    # y_pred = cross_val_predict(blender, x_test[BLENDER_FEATURES], y_test, cv = 5)
    # y_pred_score = cross_val_score(blender, x_test[BLENDER_FEATURES], y_test, cv = 5)

    os.mkdir(output_path)
    joblib.dump(blender, os.path.join(output_path, "blender.joblib"), compress=3)

    base_path = os.path.join(output_path, "base_models")
    os.makedirs(base_path)

    for i, clf in enumerate(clfs):
        path = os.path.join(base_path, f"model{i}.joblib")
        joblib.dump(clf, path, compress=3)