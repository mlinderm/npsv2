import pickle
import pandas as pd

# training set: '/storage/yirans/npsv2/tfrecords/distance_records/dictionary_HG00096.pickle'
# validation set: '/storage/yirans/npsv2/tfrecords/distance_records/dictionary_with_tier2.pickle'

def analyze_distance():
    """This was used to print info in the distances of the variants in HG0002, specifically the max hom. ref. distance and the
    min other distance so that we can find patterns with hiogh probabilities of determininghte correct genotype
    
    Specifically, this prints examples where hom. ref and others cross and lists of each with percentiles, eg. first 
    is the lowest, second is fifth percentile, etc.
    then it also includes 98th percentile for each max hom ref and min other distance"""

    file_dir = '/storage/yirans/npsv2/tfrecords/distance_records/dictionary_with_tier2.pickle'
    with open(file_dir, 'rb') as handle:
        dic = pickle.load(handle)

    # for x in dic:
    #     print(dic[x])
    #     break

    cross_count_0 = 0
    total_0 = 0
    cross_count_1 = 0
    total_1 = 0
    for x in dic:
        crosses = False
        for y in dic[x][1]:
            if y.index(min(y)) != 0:
                crosses = True
                break
        if crosses:
            if dic[x][0] == 0:
                cross_count_0 += 1
            else:
                cross_count_1 += 1
        if dic[x][0] == 0:
            total_0 += 1
        else:
            total_1 += 1
    print('cross analysis:')
    print(cross_count_0, total_0, cross_count_1, total_1)

    max_homo_ref_dist_0 = []
    max_homo_ref_dist_1 = []
    for x in dic:
        max_ = 0
        for y in dic[x][1]:
            if y[0] > max_:
                max_ = y[0]
        if dic[x][0] == 0:
            max_homo_ref_dist_0.append(max_)
        else:
            max_homo_ref_dist_1.append(max_)
    max_homo_ref_dist_0 = sorted(max_homo_ref_dist_0)
    max_homo_ref_dist_1 = sorted(max_homo_ref_dist_1)

    perc_0 = []
    perc_1 = []
    len0 = len(max_homo_ref_dist_0) - 1
    len1 = len(max_homo_ref_dist_1) - 1
    for x in range(21):
        percentile = x * 0.05
        perc_0.append(max_homo_ref_dist_0[int(len0 * percentile)])
        perc_1.append(max_homo_ref_dist_1[int(len1 * percentile)])
    print('max homo ref dist analysis:')
    print(perc_0)
    print(perc_1)
    print(max_homo_ref_dist_0[int(len0 * 0.98)])

    min_var_dist_0 = []
    min_var_dist_1 = []
    for x in dic:
        min_ = float('inf')
        for y in dic[x][1]:
            if y[1] < min_:
                min_ = y[1]
            if y[2] < min_:
                min_ = y[2]
        if dic[x][0] == 0:
            min_var_dist_0.append(min_)
        else:
            min_var_dist_1.append(min_)
    min_var_dist_0 = sorted(min_var_dist_0)
    min_var_dist_1 = sorted(min_var_dist_1)

    perc_0 = []
    perc_1 = []
    # len0 = len(min_var_dist_0) - 1
    # len1 = len(min_var_dist_1) - 1
    for x in range(21):
        percentile = x * 0.05
        perc_0.append(min_var_dist_0[int(len0 * percentile)])
        perc_1.append(min_var_dist_1[int(len1 * percentile)])
    print('min var dist analysis:')
    print(perc_0)
    print(perc_1)
    print(min_var_dist_1[int(len1 * 0.98)])

    new_dic = {}
    for x in dic:
        max_homo = 0
        min_var = float('inf')
        for y in dic[x][1]:
            if y[0] > max_homo:
                max_homo = y[0]
            if y[1] < min_var:
                min_var = y[1]
            if y[2] < min_var:
                min_var = y[2]
        new_dic[x] = (max_homo, min_var)

    zeroes = 0
    ones = 0
    true_zeroes = 0
    true_ones = 0
    undecided = 0
    for x in new_dic:
        if new_dic[x][0] > 0.75:
            if dic[x][0] == 1:
                true_ones += 1
            ones += 1
        elif new_dic[x][1] > 0.75:
            if dic[x][0] == 0:
                true_zeroes += 1
            zeroes += 1
        else:
            undecided += 1

    print(zeroes, ones, true_zeroes, true_ones, undecided)

def analyze_confidence():
    """This was used to print info on the confidences of the model predictions to see if 
    more confidence was correlated with more correct predictions"""
    
    file = pd.read_csv('results_medium', sep='\t', 
                        dtype = {'prob':float, 'pred':int, 'true':int, 'id':str, 'sv_len':int, 'pred_pos':int, 'pb_pos':str, 'matching':str})
    zeroes = []
    ones = []
    for _, row in file.iterrows():
        if row['true'] == 0:
            zeroes.append(row['prob'])
        else:
            ones.append(row['prob'])
    zeroes = sorted(zeroes)
    ones = sorted(ones)

    perc_0 = []
    perc_1 = []
    len0 = len(zeroes) - 1
    len1 = len(ones) - 1
    for x in range(21):
        percentile = x * 0.05
        perc_0.append(zeroes[int(len0 * percentile)])
        perc_1.append(ones[int(len1 * percentile)])
    print('confidence analysis:')
    print(perc_0)
    print(perc_1)
    threshold_1 = zeroes[int(len0 * 0.98)]
    threshold_0 = ones[int(len1 * 0.02)]
    print(threshold_1)
    print(threshold_0)

    threshold_0 = 0.009
    threshold_1 = 0.98

    ones_ = 0
    zeroes_ = 0
    true_ones = 0
    true_zeroes = 0
    undecided = 0
    for x in ones:
        if x >= threshold_1:
            ones_ += 1
            true_ones += 1
        elif x <= threshold_0:
            zeroes_ += 1
        else:
            undecided += 1
    for x in zeroes:
        if x >= threshold_1:
            ones_ += 1
        elif x <= threshold_0:
            zeroes_ += 1
            true_zeroes += 1
        else:
            undecided += 1
    print(ones_, true_ones, zeroes_, true_zeroes, undecided)

def string_to_float_list(string):
    """Converts a string that looks like a list of lists of floats into a list of lists of floats
    does not return a list if input string is '.' """
    if string == '.':
        return '.'
    return [float(i) for i in (string.split(','))]

def evaluate_process():
    """Evaluates the distance model's ability as a refiner, end by printing correct before and after"""

    file = pd.read_csv('test5.tsv', sep='\t', 
                        dtype = {'ID':str, 'POS':int, 'SVLEN':int, 'GIAB_GT':str, 'MatchGT':bool, 'GT':str, 'OGT':str, 'ALTS':str, 'CL':str},
                        converters = {'DS':string_to_float_list, 'ODS':string_to_float_list})
    file2 = pd.read_csv('results_test', sep='\t', 
                        dtype = {'prob':float, 'pred':int, 'true':int, 'id':str, 'sv_len':int, 'pred_pos':int, 'pb_pos':str, 'matching':str})
    file2 = file2.drop(columns = ['prob', 'true', 'sv_len', 'pred_pos', 'pb_pos', 'matching', 'likelihood'])
    file3 = pd.read_csv('results_test2', sep='\t', 
                        names = ['id', 'pred'], dtype = {'id':str, 'pred':int})
    file2 = file2.append(file3, ignore_index=True)
    dic = {}
    f = open('new_false_negatives', 'w')
    g = open('new_true_negatives', 'w')
    for _, row in file2.iterrows():
        dic[row['id']] = row['pred']
    init_correct = 0
    new_correct = 0
    corrected = 0
    for _, row in file.iterrows():
        if row['GIAB_GT'] == row['GT']:
            init_correct += 1
        if dic[row['ID']] == 0 and row['OGT'] != '.' and row['GT'] != '0/0':
            corrected += 1
            gt = row['OGT']
        else:
            gt = row['GT']
        if row['GIAB_GT'] == gt:
            new_correct += 1
        if row['GIAB_GT'] != gt and row['GIAB_GT'] == row['GT']:
            f.write(row['ID'] + '\n')
        if row['GIAB_GT'] == gt and row['GIAB_GT'] != row['GT']:
            g.write(row['ID'] + '\n')
    print('correct before: ' + str(init_correct) + ' correct after: ' + str(new_correct))
    #print(corrected)

if __name__ == '__main__':
    #analyze_confidence()
    evaluate_process()

# training:
# cross analysis:3702 25323 7007 7987
# max homo ref dist analysis:
# [0.0107, 0.0275, 0.0337, 0.0395, 0.0457, 0.0529, 0.0611, 0.0701, 0.0795, 0.0897, 0.1001, 0.1118, 0.1251, 0.1421, 0.1665, 0.2093, 0.2811, 0.3713, 0.485, 0.625, 1.0052]
# [0.0175, 0.1486, 0.3409, 0.4616, 0.5415, 0.6145, 0.6752, 0.7241, 0.7626, 0.7963, 0.8251, 0.8481, 0.8661, 0.8825, 0.8977, 0.9135, 0.9299, 0.9477, 0.9653, 0.9816, 1.0242]
# 0.7681
# min var dist analysis:
# [0.0099, 0.1203, 0.2217, 0.3282, 0.418, 0.4858, 0.5425, 0.5913, 0.6316, 0.6656, 0.6992, 0.7313, 0.7663, 0.8081, 0.8535, 0.8812, 0.8978, 0.9109, 0.9223, 0.9365, 0.9722]
# [0.0054, 0.0229, 0.0281, 0.0332, 0.0382, 0.0433, 0.0492, 0.0558, 0.0633, 0.0726, 0.0839, 0.1001, 0.118, 0.1425, 0.1771, 0.2207, 0.2788, 0.3494, 0.4466, 0.6049, 0.9555]
# 0.7755
# 10868 5482 10682 4906 16960

# val:
# cross analysis:
# 1235 4906 7811 8216
# max homo ref dist analysis:
# [0.0059, 0.0522, 0.0757, 0.0946, 0.1106, 0.1289, 0.1442, 0.1651, 0.1906, 0.2138, 0.2405, 0.2695, 0.3015, 0.3325, 0.3667, 0.4092, 0.4591, 0.5232, 0.6004, 0.7022, 0.9312]
# [0.0194, 0.4013, 0.5149, 0.5911, 0.6443, 0.6865, 0.7238, 0.7548, 0.7853, 0.8118, 0.8337, 0.8515, 0.8674, 0.8813, 0.8927, 0.9034, 0.9142, 0.9256, 0.9375, 0.9548, 1.0659]
# 0.7966
# min var dist analysis:
# [0.0233, 0.1121, 0.177, 0.2476, 0.3106, 0.3573, 0.405, 0.4426, 0.4759, 0.5099, 0.5372, 0.5639, 0.589, 0.6126, 0.6386, 0.6618, 0.684, 0.7089, 0.7366, 0.8218, 0.9697]
# [0.0063, 0.0158, 0.0195, 0.0229, 0.0266, 0.0302, 0.0342, 0.0391, 0.0451, 0.0526, 0.0616, 0.0721, 0.0842, 0.0991, 0.1211, 0.1474, 0.1788, 0.2251, 0.2816, 0.3715, 0.9151]
# 0.4801
# 416 5573 408 5418 7133
