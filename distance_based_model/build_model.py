import tensorflow as tf
import os
from npsv2 import npsv2_pb2
from google.protobuf import descriptor_pb2
from npsv2 import images
import pandas as pd
import typing
import pickle
from sklearn.metrics import mean_squared_error as mse

def _bytes_feature(list_of_strings):
    """Returns a bytes_list from a list of string / byte."""
    if isinstance(list_of_strings, type(tf.constant(0))):
        list_of_strings = list_of_strings.numpy() # BytesList won't unpack a string from an EagerTensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[list_of_strings]))

def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(list_of_floats):
    """Returns a float_list from a list of int / bool."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[list_of_floats]))

def create_example(list_, name, SV_length):
    """Takes a list containing class, list of distances, location, start, and end as well as a name and sv length for this variant
    Returns a tensor containing the same information"""

    length = len(list_[1])
    list_[1] = tf.convert_to_tensor(list_[1], dtype=tf.float32)
    list_[1] = tf.ensure_shape(list_[1], [length, 3])
    list_[1] = tf.image.resize(tf.expand_dims(list_[1], axis=-1), (512, 3))
    list_[1] = tf.squeeze(list_[1], axis=-1)

    name = tf.convert_to_tensor(name, dtype=tf.string)
    SV_length = tf.convert_to_tensor(SV_length, dtype=tf.int64)

    feature = {
        'id': _bytes_feature(name),
        'SV_length': int64_feature(SV_length),
        'class': int64_feature(list_[0]),
        'location': _float_feature(list_[2]),
        #'distances/shape': int64_feature(len(list_[1])),
        'distances/encoded': _bytes_feature(tf.io.serialize_tensor(list_[1])),
        'start': int64_feature(list_[3]),
        'end': int64_feature(list_[4]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_records(x_train, tfrecords_dir, name):
    """Save the tfrecords file to storage for one chr file
    Args:
        x_train is the dictionary of variants to save
        tfrecords_dir is the directory that the tensor is saved to
        name is the specific file name that the tensor is save to"""

    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)  # creating TFRecords output folder

    with tf.io.TFRecordWriter(
        tfrecords_dir + name
    ) as writer:
        for feature in x_train:
            example = create_example(x_train[feature], feature[0], feature[1])
            writer.write(example.SerializeToString())

def load_dataset(filename: str, training: bool) -> tf.data.Dataset:
    """Load embeddings dataset from tfrecords file

    Args:
        filename (str): tfrecords file, can be compressed, e.g. *.tfrecords.gz
        training (bool): to avoid errors in the model, set training to True if this dataset is meant for training, 
            otherwise is you want to pull more than the labels and distances out of the tensor (like maybe in predict), set to False

    Returns:
        tf.data.Dataset: Dataset with decoded tensors and SV start (POS) and svlen (INFO/SVLEN) fields
    """

    # Description to parse SV protobuf saved with embeddings
    file_descriptor_set = descriptor_pb2.FileDescriptorSet()
    npsv2_pb2.DESCRIPTOR.CopyToProto(file_descriptor_set.file.add())
    descriptor_source = b"bytes://" + file_descriptor_set.SerializeToString()
    if not training:
        proto_features = {
            'id': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'SV_length': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'class': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'distances/encoded': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            #'distances/shape': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'location': tf.io.FixedLenFeature(shape=(), dtype=tf.float32),
            'start': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'end': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        }
    else:
        proto_features = {
            'class': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'distances/encoded': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            #'distances/shape': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'location': tf.io.FixedLenFeature(shape=(), dtype=tf.float32),
        }

    def _process_input(proto_string):
        """Helper function for input function that parses a serialized example."""

        parsed_features = tf.io.parse_single_example(serialized=proto_string, features=proto_features)

        features = {
            "distances": tf.io.parse_tensor(parsed_features["distances/encoded"], tf.float32),
        }
        if not training:
            labels = {
                "class": parsed_features["class"],
                "location": parsed_features["location"],
                "id": parsed_features["id"],
                "SV_length": parsed_features["SV_length"],
                "start": parsed_features["start"],
                "end": parsed_features["end"],
            }
        else:
            labels = {
                "class": parsed_features["class"],
                "location": parsed_features["location"],
            }

        return features, labels

    compression = images._filename_to_compression(filename)
    return tf.data.TFRecordDataset(filename, compression_type=compression).map(_process_input)

def string_to_float_list(string):
    """Converts a string that looks like a list of lists of floats into a list of lists of floats
    does not return a list if input string is '.' """
    if string == '.':
        return '.'
    return [float(i) for i in (string.split(','))]

def _genotype_to_label(genotype): # , alleles: typing.AbstractSet[int]={1,2}
    # TODO: Handle no-calls
    """Takes a string like '0/0' or '0|2' and return 0 if that string means hom. ref., but otherise returns 1"""
    for x in genotype:
        if x == '1' or x == '2':
            return 1
    return 0

#create the model
def convolutional_block(inputs):
    """Defines the convolutional block for the neural network with four layers"""

    x = tf.keras.layers.Conv1D(16, 3, padding = 'same', activation = 'relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    
    x = tf.keras.layers.Conv1D(32, 3, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    
    x = tf.keras.layers.Conv1D(64, 5, padding = 'valid', activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
        
    x = tf.keras.layers.Conv1D(64, 5, padding = 'valid', activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    
    return x

def regression_block(x):
    """Defines the regression block (block that predicts the location of the variant) for the neural network"""
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    x = tf.keras.layers.Dense(1, name = 'location')(x)
    
    return x

def classification_block(x):
    """Defines the classification block for the neural network"""
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    x = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'class')(x)
    
    return x

def make_dict_2():
    """Makes and saves a dictionary for HG00096 at /storage/yirans/npsv2/tfrecords/distance_records/dictionary_HG00096.pickle"""

    #deal with originals 

    file_ = '/storage/yirans/npsv2/distance_tables/'
    chroms = ['chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',  'chr9']
    table = pd.read_csv(file_ + 'chr10', sep='\t',
                        names = ['variant_ID', 'original_ID', 'SV_length', 'chromosome', 'position', 'predicted_genotype', 'distances'], 
                        dtype = {'variant_ID':str, 'original_ID':str, 'SV_length':int, 'chromosome':str, 'position':int, 'predicted_genotype':str}, 
                        converters = {'distances':string_to_float_list})
    #print(table)
    for x in chroms:
        new = pd.read_csv(file_ + x, sep='\t',
                        names = ['variant_ID', 'original_ID', 'SV_length', 'chromosome', 'position', 'predicted_genotype', 'distances'], 
                        dtype = {'variant_ID':str, 'original_ID':str, 'SV_length':int, 'chromosome':str, 'position':int, 'predicted_genotype':str}, 
                        converters = {'distances':string_to_float_list})
        table = pd.concat([table, new], ignore_index=True)
    labels = pd.read_csv(file_ + 'labels_table', sep='\t',
                    names = ['ID', 'SV_length', 'chromosome', 'position', 'true_genotype'],
                    dtype = {'ID':str, 'SV_length':int, 'chromosome':str, 'position':int, 'true_genotype':str})
    #labels = labels.drop(columns = ['ID'])
    originals = table[table.variant_ID != '.']
    labels = labels.merge(originals, on=['chromosome', 'position', 'SV_length'], how='inner')
    table = table.sort_values(by = ['chromosome', 'position'], ascending = [True, True])

    dic = {}
    original_dic = {}
    genotype_dic = {}
    for _, row in labels.iterrows():
        genotype_dic[row['variant_ID']] = _genotype_to_label(row['true_genotype'])
    for _, row in table.iterrows():
        if row['variant_ID'] == '.':
            id = row['original_ID']
        else:
            id = row['variant_ID']
            original_dic[id] = row['position']
        sv_len = row['SV_length']
        if id not in genotype_dic:
            continue
        if (id, sv_len) not in dic:
            dic[(id, sv_len)] = [genotype_dic[id], [], [], [], []]
        dic[(id, sv_len)][1].append(row['distances'])
        dic[(id, sv_len)][3].append(row['position'])

    for x in dic:
        high = dic[x][3][-1] 
        low = dic[x][3][0]
        orig = original_dic[x[0]]
        if high != low:
            dic[x][2] = (orig - low) / (high - low)
        else:
            dic[x][2] = 0.0
        dic[x][3] = low
        dic[x][4] = high 

    # for x in dic:
    #     print(dic[x])
    #     break
    
    with open('/storage/yirans/npsv2/tfrecords/distance_records/dictionary_HG00096.pickle', 'wb') as handle:
        pickle.dump(dic, handle)

def make_dict():
    """makes and saves a dictionary for HG0002 tiers 1 and 2 at /storage/yirans/npsv2/tfrecords/distance_records/dictionary_with_tier2.pickle"""

    file1 = pd.read_csv('/storage/mlinderman/projects/sv/npsv2-experiments/proposals.tsv', sep='\t', 
                        names = ['variant_ID', 'original_ID', 'SV_length', 'chromosome', 'position', 'predicted_genotype', 'distances'], 
                        dtype = {'variant_ID':str, 'original_ID':str, 'SV_length':int, 'chromosome':str, 'position':int, 'predicted_genotype':str}, 
                        converters = {'distances':string_to_float_list})
    file2 = pd.read_csv('/storage/mlinderman/projects/sv/npsv2-experiments/concordance.csv', sep=',', 
                        names = ['ID', 'chromosome', 'position', 'SV_length', 'true_genotype', 'matching', 'predicted_genotype'], 
                        dtype = {'ID':str, 'SV_length':int, 'chromosome':str, 'position':int, 'predicted_genotype':str, 'true_genotype':str, 'matching':bool})
    file1 = file1.sort_values(by = ['chromosome', 'position'], ascending = [True, True])
    # print(file1)
    
    dic = {}
    original_dic = {}
    genotype_dic = {}
    for _, row in file2.iterrows():
        genotype_dic[row['ID']] = _genotype_to_label(row['true_genotype'])
    for _, row in file1.iterrows():
        if row['variant_ID'] == '.':
            id = row['original_ID']
        else:
            id = row['variant_ID']
            original_dic[id] = row['position']
        sv_len = row['SV_length']
        if id not in genotype_dic:
            continue
        if (id, sv_len) not in dic:
            dic[(id, sv_len)] = [genotype_dic[id], [], [], [], []]
        dic[(id, sv_len)][1].append(row['distances'])
        dic[(id, sv_len)][3].append(row['position'])

    for x in dic:
        high = dic[x][3][-1] 
        low = dic[x][3][0]
        orig = original_dic[x[0]]
        if high != low:
            dic[x][2] = (orig - low) / (high - low)
        else:
            dic[x][2] = 0.0
        dic[x][3] = low
        dic[x][4] = high 

    with open('/storage/yirans/npsv2/tfrecords/distance_records/dictionary_with_tier2.pickle', 'wb') as handle:
        pickle.dump(dic, handle)

def make_tfrec(file_dir, save_dir, filter):
    """converts a dictionary from file_dir to a tfrecord file (save to save_dir) 
    and optionally applies a filter for easy examples"""

    # training set: '/storage/yirans/npsv2/tfrecords/distance_records/dictionary_HG00096.pickle'
    # validation set: '/storage/yirans/npsv2/tfrecords/distance_records/dictionary_with_tier2.pickle'
    with open(file_dir, 'rb') as handle:
        dic = pickle.load(handle)

    # file = pd.read_csv('test5.tsv', sep='\t', 
    #                     dtype = {'ID':str, 'POS':int, 'SVLEN':int, 'GIAB_GT':str, 'MatchGT':bool, 'GT':str, 'OGT':str, 'ALTS':str, 'CL':str},
    #                     converters = {'DS':string_to_float_list, 'ODS':string_to_float_list})
    # ids = []
    # for _, row in file.iterrows():
    #     ids.append(row['ID'])
    # dic_copy = dic.copy()
    # for x, y in dic:
    #     if x not in ids:
    #         del dic_copy[(x, y)]
    # dic = dic_copy
    if filter:
        new_dic = {}            # use this for filter
        for id, sv_len in dic:
            max_homo = 0
            min_var = float('inf')
            for y in dic[id, sv_len][1]:
                if y[0] > max_homo:
                    max_homo = y[0]
                if y[1] < min_var:
                    min_var = y[1]
                if y[2] < min_var:
                    min_var = y[2]
            if id not in new_dic:
                new_dic[id] = [[sv_len], max_homo, min_var]
            else:
                new_dic[id][0].append(sv_len)
                new_dic[id][1] = max(new_dic[id][1], max_homo)
                new_dic[id][2] = max(new_dic[id][2], min_var)
        print('total variants: ' + str(len(new_dic)))

        zeroes = 0
        true_zeroes = 0
        ones = 0
        true_ones = 0
        undecided = 0
        f = open('results_test2', 'w')
        for id in new_dic:
            if (new_dic[id][1] > 0.75) ^ (new_dic[id][2] > 0.75):
                if new_dic[id][1] > 0.75:
                    if dic[(id, new_dic[id][0][0])][0] == 1:
                        true_ones += 1
                    ones += 1
                    f.write(id + '\t1\n')
                else:
                    if dic[(id, new_dic[id][0][0])][0] == 0:
                        true_zeroes += 1
                    zeroes += 1
                    f.write(id + '\t0\n')
                for sv_len in new_dic[id][0]:
                    del dic[id, sv_len]
            else:
                undecided += 1
        print(zeroes, true_zeroes, ones, true_ones, undecided)

    tfrecords_dir = '/storage/yirans/npsv2/tfrecords/distance_records'
    write_records(dic, tfrecords_dir, save_dir)
    # training save: "/HG00096.tfrec"
    # val save: "/all_with_tier2.tfrec"

def make_model():
    """make model defines a model, loads in data, trains on said data, and saves the model
    options such as training and validation data file names, epochs, and under what name the model is saved all must be changed manually below"""

    save_dir = './saved_models/medium_examples_model2'
    file_dir = '/storage/yirans/npsv2/tfrecords/distance_records/HG00096_medium_examples_training.tfrec'
    val_dir = '/storage/yirans/npsv2/tfrecords/distance_records/tier2_medium_examples.tfrec'
    epoch_num = 40

    # define model
    inputs = tf.keras.Input((512, 3),
                            name="distances")
    x = convolutional_block(inputs)
    box_output = regression_block(x)
    class_output = classification_block(x)
    model = tf.keras.Model(inputs = inputs, outputs = [class_output, box_output])
    model.summary()

    # load data
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
                loss={'class': 'binary_focal_crossentropy', 'location': 'mse'},
                metrics={'class': 'accuracy', 'location': 'mse'})

    #training, validation = get_dataset_partitions(load_dataset(file_dir).batch(32))

    training = load_dataset(file_dir, True).batch(32)
    validation = load_dataset(val_dir, True).batch(32)

    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #                             filepath='./saved_models/20epoch_HG00096',
    #                             monitor='val_accuracy',
    #                             mode='max',
    #                             save_best_only=True)

    model.fit(x=training, validation_data=validation, epochs=epoch_num) # callbacks=[model_checkpoint_callback] 
    model.save(save_dir)

#https://towardsdatascience.com/how-to-split-a-tensorflow-dataset-into-train-validation-and-test-sets-526c8dd29438
def get_dataset_partitions(ds, train_split=0.9):
    """Given a loaded dataset (ds) and train_split (a float between 0 and 1), will return the dataset after being partitioned 
    eg. 90%, 10% (if train_split = 0.9 
    this is useful for creating validation data out of a subset of the training data"""

    assert train_split < 1
    
    split_key = int(1 / (1 - train_split))

    def split_train_fn(dataset):
        # returns the training portion of the dataset
        dataset = (
            dataset.enumerate()
            .filter(lambda x,_: x % split_key != 0)
            .map(lambda _,y: y)
        )
        return dataset
    
    def split_val_fn(dataset):
        # returns the validation portion of the dataset
        dataset = (
            dataset.enumerate()
            .filter(lambda x,_: x % split_key == 0)
            .map(lambda _,y: y)
        )
        return dataset
    
    return split_train_fn(ds), split_val_fn(ds)

def eval_model():
    """saves the results of prediction into a file and prints accuracy"""

    results_dir = 'results_test'
    file_dir = '/storage/yirans/npsv2/tfrecords/distance_records/tier2_test.tfrec'
    model_path = './saved_models/medium_examples_model'

    validation = load_dataset(file_dir, False).batch(1)
    true = []
    ids = []
    starts = []
    ends = []
    sv_lens = []
    for a in validation:
        for b in a[1]['class']:
            true.append(b.numpy())
        for b in a[1]['id']:
            ids.append(b.numpy().decode())
        for start in a[1]['start']:
            starts.append(start.numpy())
        for end in a[1]['end']:
            ends.append(end.numpy())
        for sv_len in a[1]['SV_length']:
            sv_lens.append(sv_len.numpy())
    m = tf.keras.models.load_model(model_path)
    pred = m.predict(x=validation)
    arrpred = []
    for x in pred[0]:
        for y in x:
            arrpred.append(y)

    locations = []
    count = 0
    for x in pred[1]:
        for y in x:
            location = round(((ends[count] - starts[count]) * y) + starts[count])
            locations.append(location)
            count += 1

    with open('/storage/yirans/npsv2/tfrecords/distance_records/pb_pos.pickle', 'rb') as handle:
        dic = pickle.load(handle)

    f = open(results_dir, 'w')
    right = 0
    wrong = 0
    f.write('prob\tpred\ttrue\tid\tsv_len\tpred_pos\tpb_pos\tmatching\n')
    for i in range(len(true)):
        f.write(str(arrpred[i]) + '\t')
        f.write(str(round(arrpred[i])) + '\t')
        f.write(str(true[i]) + '\t')
        f.write(ids[i] + '\t')
        f.write(str(sv_lens[i]) + '\t')
        f.write(str(locations[i]) + '\t')
        if ids[i] in dic:
            f.write(str(dic[ids[i]]) + '\t')
        else:
            f.write('None\t')
        if round(arrpred[i]) != true[i]:
            f.write('!\n')
            wrong += 1
        else:
            f.write(' \n')
            right += 1

    f.close()

    #print(right / (right + wrong))
    condense_results(results_dir)

def condense_results(results_dir):
    """This function alters the results from eval_model so that only the most sure (prob is closest to 0 or 1) sv_length 
    remains for each variant"""

    file = pd.read_csv(results_dir, sep='\t', 
                        dtype = {'prob':float, 'pred':int, 'true':int, 'id':str, 'sv_len':int, 'pred_pos':int, 'pb_pos':str, 'matching':str})

    new_col = []
    for _, row in file.iterrows():
        new_col.append(abs(row['prob'] - 0.5))
    file['likelihood'] = new_col

    file = file.sort_values(by=['id', 'likelihood'], ascending=[True, True], ignore_index=True)
    file = file.drop_duplicates(subset=['id'], keep='first')
    file.to_csv(path_or_buf = results_dir, sep = '\t', index = False)

    correct = 0
    total = 0
    for _, row in file.iterrows():
        if row['pred'] == row['true']:
            correct += 1
        total += 1
    print (total, correct / total)

def make_position_dic():
    """makes and saves a dictionary of the true locations based on truvari for HG002"""

    # file1 = pd.read_csv('/storage/mlinderman/projects/sv/npsv2-experiments/concordance.csv', sep=',', 
    #                     names = ['ID', 'chromosome', 'position', 'SV_length', 'true_genotype', 'matching', 'predicted_genotype'], 
    #                     dtype = {'ID':str, 'SV_length':int, 'chromosome':str, 'position':int, 'predicted_genotype':str, 'true_genotype':str, 'matching':bool})
    
    # dic = {}
    # for _, row in file1.iterrows():
    #     dic[row['ID']] = [row['SV_length'], row['position'], []]

    file = pd.read_csv('/storage/mlinderman/projects/sv/npsv2-experiments/resources/truvari_HG002_pbsv_DEL.pos.csv', 
                        dtype = {'MatchID':int, 'ID':str, 'GIAB_GT':str, 'CHROM':str, 'GIAB_POS':int, 'GIAB_SVLEN':int, 'PB_POS':int, 'PB_SVLEN':int})
    dic = {}
    for _, row in file.iterrows():
        dic[row['ID']] = row['PB_POS']

    with open('/storage/yirans/npsv2/tfrecords/distance_records/pb_pos.pickle', 'wb') as handle:
        pickle.dump(dic, handle)

if __name__ == '__main__':

    #make_position_dic()
    #make_dict()
    #make_dict_2()
    #make_tfrec('/storage/yirans/npsv2/tfrecords/distance_records/dictionary_with_tier2.pickle', '/all_with_tier2.tfrec', False)
    #make_tfrec('/storage/yirans/npsv2/tfrecords/distance_records/dictionary_HG00096.pickle', '/HG00096.tfrec', False)

    # make_tfrec('/storage/yirans/npsv2/tfrecords/distance_records/dictionary_HG00096.pickle', '/HG00096_medium_examples_training.tfrec', False)
    # make_tfrec('/storage/yirans/npsv2/tfrecords/distance_records/dictionary_with_tier2.pickle', '/tier2_only_wrong_calls.tfrec', False)
    # make_model()

    #make_tfrec('/storage/yirans/npsv2/tfrecords/distance_records/dictionary_with_tier2.pickle', '/tier2_medium_examples.tfrec', False)

    #make_tfrec('/storage/yirans/npsv2/tfrecords/distance_records/dictionary_with_tier2.pickle', '/tier2_test.tfrec', True)
    eval_model()