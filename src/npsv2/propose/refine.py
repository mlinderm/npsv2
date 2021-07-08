import joblib, operator
import pysam
import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm
from ..variant import Variant
from ..range import Range, RangeTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from imblearn.under_sampling import RandomUnderSampler
import joblib

ORIGINAL_KEY = "ORIGINAL"

FEATURES = ["HOMO_REF_DIST","HET_DIST","HOMO_ALT_DIST","DHFFC","MIN","DIFF","BIG CONF","SMALL CONF"]

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
            rows.append(pd.DataFrame({
                "ID": record.id,
                "POS": int(record.pos),
                "END": record.stop,
                "ORIGINAL": original,
                "SV": original or record.id,
                "SAMPLE": i,
                "GT":  "/".join(map(str, call.allele_indices)),
                "ORIGINAL_MIN": orig_min_dist[i],
                "HOMO_REF_DIST": distances[0],
                "HET_DIST": distances[1],
                "HOMO_ALT_DIST": distances[2],
                "DHFFC": call["DHFFC"],
            },index=[0]))
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

def _vcf_to_table(vcf_path):
    with pysam.VariantFile(vcf_path) as src_vcf_file:
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

def refine_vcf(cfg, vcf_path: str, output_path: str, progress_bar=False, include_orig_ref=True, merge_blocks=True, include_orig_in_block=False, classifier_path=None):
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

        # original_records = {}
        # alternate_records = {}

        # for record in src_vcf_file:
        #     if ORIGINAL_KEY not in record.info:
        #         assert record.id and record.id not in original_records, "Duplicate original variants"
        #         original_records[record.id] = record
        #     else:
        #         originals = record.info[ORIGINAL_KEY]
        #         for original in originals:
        #             if original in alternate_records:
        #                 alternate_records[original].append(record)
        #             else:
        #                 alternate_records[original] = [record]

        
        # rows = []
        # for id, original_record in original_records.items():
        #     # Determine minimum "original" distance for each sample
        #     orig_min_dist = [np.min(call["DS"]) for call in original_record.samples.itervalues()]

        #     rows.append(_record_to_rows(original_record, orig_min_dist))
        #     for alt_record in alternate_records.get(id, []):
        #         rows.append(_record_to_rows(alt_record, orig_min_dist))
        # table = pd.concat(rows, ignore_index=True)

        # _add_derived_features(table)
        table, original_records, alternate_records = _vcf_to_table(vcf_path)

        # Load the "refine" classifier
        clf = joblib.load(classifier_path)
        table["PROBTRUE"] = clf.predict_proba(table[FEATURES])[:,1]

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
                
                for i, call in enumerate(record.samples.itervalues()):
                    possible_calls = variant_table.get_group((id, i)).reset_index(drop=True)

                    # with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
                    #     print(possible_calls)

                    if possible_calls.shape[0] == 1:
                        # No alternate record to update with
                        assert pd.isna(possible_calls.loc[possible_calls.index[0],"ORIGINAL"])
                        continue
                    else:
                        #min_dist_idx = np.unravel_index(np.argmin(possible_calls[["HOMO_REF_DIST","HET_DIST","HOMO_ALT_DIST"]]), (possible_calls.shape[0], 3))
                        #alt_row = possible_calls.iloc[min_dist_idx[0], :]
                        
                        max_prob_idx = possible_calls.PROBTRUE.idxmax()
                        alt_row = possible_calls.iloc[max_prob_idx, :]
                        if not pd.isna(alt_row.ORIGINAL):
                            call.update({
                                "DS": [alt_row.HOMO_REF_DIST, alt_row.HET_DIST, alt_row.HOMO_ALT_DIST],
                                "DHFFC": alt_row.DHFFC,
                                "CL": f"{alt_row.POS}_{alt_row.END}",
                            })
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

def train_model(vcf_path: str, pbsv_file: str, output_path: str):
    #create table to train model and return saved model
    table, *_ = _vcf_to_table(vcf_path)
    
    FEATURES_VALS = ["HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST", "DHFFC", "BIG CONF", "SMALL CONF", "MIN", "DIFF"]
    FEATURES = ["ID", "ORIGINAL", "GT", "POS", "END", "HOMO_REF_DIST", "HET_DIST", "HOMO_ALT_DIST", "DHFFC", "BIG CONF", "SMALL CONF", "MIN", "DIFF", "SV"]

    pbsv_table = pd.read_csv(pbsv_file, sep='\t')
    pbsv_table.columns = ['ID', 'ORIGINAL', 'POS', 'END', 'MATCHGT']

    merge_table_prop = pd.merge(table, pbsv_table, on=["ORIGINAL","POS","END"], how = "left")
    merge_table_prop = merge_table_prop.drop('ID_y', axis=1)
    merge_table_prop = merge_table_prop.rename(columns={'ID_x':'ID'})

    merge_table_original = pd.merge(table, pbsv_table, on=["ID","POS","END"], how = "left")
    merge_table_original = merge_table_original.drop('ORIGINAL_y', axis=1)
    merge_table_original = merge_table_original.rename(columns={'ORIGINAL_x':'ORIGINAL'})

    train_table = pd.merge(merge_table_original, merge_table_prop, on = FEATURES)
    train_table['PBSV'] = (train_table.MATCHGT_x == True) | (train_table.MATCHGT_y == True)
    train_table = train_table.drop(columns=['MATCHGT_x','MATCHGT_y'], axis=1)

    train_sorted = train_table.sort_values(by = 'SV').reset_index()
    train_input = train_sorted[FEATURES_VALS]
    train_output = train_sorted['PBSV']

    #our own undersample, includes a false for every true with at least one false
    true_table = train_table[train_table["PBSV"] == True]
    undersampled_df = true_table
    zeros = 0
    for sv in true_table['SV']:
        filtered_table = train_table[train_table['SV'] == sv]
        filtered_table = filtered_table[filtered_table['PBSV'] == False]
        if(len(filtered_table.index) > 0):
            row = filtered_table.iloc[0]
            undersampled_df = undersampled_df.append(row)
        else:
            zeros += 1

    merge_with_sample = pd.merge(train_table, undersampled_df, how = "left", on = train_table.columns.values.tolist(), indicator = "EXIST")
    unsampled_false_table = merge_with_sample[merge_with_sample['EXIST'] == 'left_only']
    false_samples = unsampled_false_table.sample(n = zeros)
    undersampled_df = pd.concat([undersampled_df, false_samples])

    #first training to select hard falses
    clf = RandomForestClassifier(n_estimators = 100, class_weight = "balanced")
    clf.fit(undersampled_df[FEATURES_VALS], undersampled_df['PBSV'])
    train_sorted.loc[:,'PROBFALSE'] = (clf.predict_proba(train_sorted[FEATURES_VALS]))[:, 0]
    train_df_true = train_sorted[train_sorted['PBSV'] == True]
    train_df_false = train_sorted[train_sorted['PBSV'] == False]
    train_df_hardfalse = train_df_false.groupby('SV').min('PROBFALSE')
    train_df = pd.concat([train_df_true, train_df_hardfalse])

    hardfalse_train_input = train_df[FEATURES]
    hardfalse_train_output = train_df['PBSV']

    #second random undersampling for hard falses
    rus = RandomUnderSampler()
    x_res, y_res = rus.fit_resample(hardfalse_train_input, hardfalse_train_output)
    undersampled_hardfalse = x_res.merge(y_res, left_index = True, right_index = True)

    #train on hard falses
    undersampled_rand = undersampled_hardfalse.sample(frac=1)
    clf = RandomForestClassifier(n_estimators = 100, class_weight = "balanced")
    clf.fit(undersampled_rand[FEATURES_VALS], undersampled_rand['PBSV'])

    #save model
    joblib.dump(clf, output_path, compress=3)



