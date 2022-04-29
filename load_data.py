import pandas as pd
import numpy as np
import os
import json
from feature_extraction.extract_features_functional import get_annotations


arouse_path = os.path.relpath('../MSP Data/Annotations/Arousal')
dominance_path = os.path.relpath('../MSP Data/Annotations/Dominance')
valence_path = os.path.relpath('../MSP Data/Annotations/Valence')
train_path = os.path.relpath('../Data/train')
validation_path = os.path.relpath('../Data/validation')
test_path = os.path.relpath('../Data/test')
data_path = os.path.relpath('../Data')


def load_features(path_features, num_inst, max_seq_len):
    # Check for header and separator
    print(path_features + os.listdir(path_features)[0])
    file = path_features + os.listdir(path_features)[0]
    with open(file) as infile:
        line = infile.readline()

    sep = ';'
    if ',' in line:
        sep = ','
    files = os.listdir(path_features)
    files.sort()
    # Read feature files
    num_features = len(pd.read_csv(path_features + os.listdir(path_features)[1], sep=sep).columns) - 2  # do not consider instance name and time stamp
    features = np.empty((num_inst, max_seq_len, num_features))
    for n, file in enumerate(files):
        F = pd.read_csv(path_features + file, sep=sep,
                        usecols=range(2, 2 + num_features)).values
        if F.shape[0] > max_seq_len: F = F[:max_seq_len, :]  # might occur for some feature representations
        features[n, :, :] = np.concatenate((F, np.zeros((max_seq_len - F.shape[0], num_features))))  # zero padded
    return features


def load_labels(path_labels, partition, num_inst, max_seq_len, targets):
    labels_original = []
    labels_padded   = []
    for tar in targets:
        labels_original_tar = []
        labels_padded_tar   = np.empty((num_inst, max_seq_len, 1))
        for n in range(0, num_inst):
            yn = pd.read_csv(path_labels + partition + '_' + str(n+1).zfill(2) + '.csv', sep=';', usecols=[tar]).values
            labels_original_tar.append(yn)  # original length sequence
            labels_padded_tar[n,:,:] = np.concatenate((yn, np.zeros((max_seq_len - yn.shape[0], 1))))  # zero padded
        labels_original.append(labels_original_tar)
        labels_padded.append(labels_padded_tar)
    return labels_original, labels_padded


def extract_features(annotations, phase=None):

    segment_path = os.path.relpath('../MSP Data/Time Labels/segments.json')
    f = open(segment_path, 'r')
    timing_data = json.load(f)
    for key in timing_data:
        conv_part = timing_data[key]['Conversation_Part']
        start = timing_data[key]['Start_Time']
        end = timing_data[key]['End_Time']
        arousal = []
        valence = []
        dominance = []
        a_annotations = annotations[0].get(conv_part)
        v_annotations = annotations[1].get(conv_part)
        d_annotations = annotations[2].get(conv_part)
        for anno in a_annotations:
            pp = os.path.join(arouse_path, anno)
            df = pd.read_csv(pp, header=8, names=['time', 'arousal'])
            df = df[(start <= df['time']) & (df['time'] <= end)]
            if df.empty:
                continue
            arousal.append(df['arousal'].mean())
        for anno in v_annotations:
            pp = os.path.join(valence_path, anno)
            df = pd.read_csv(pp, header=8, names=['time', 'valence'])
            df = df[(start <= df['time']) & (df['time'] <= end)]
            if df.empty:
                continue
            valence.append(df['valence'].mean())
        for anno in d_annotations:
            pp = os.path.join(dominance_path, anno)
            df = pd.read_csv(pp, header=8, names=['time', 'dominance'])
            if anno == 'MSP-Conversation_0047_2_001.csv':
                df = df.reset_index()
                df = df.drop(columns=['dominance'])
                df.columns = ['time', 'dominance']
            df = df[(start <= df['time']) & (df['time'] <= end)]
            if df.empty:
                continue
            dominance.append(df['dominance'].mean())

        score_arousal = mean(arousal)
        score_valence = mean(valence)
        score_dominance = mean(dominance)
        data.loc[key, 'arousal'] = score_arousal
        data.loc[key, 'valence'] = score_valence
        data.loc[key, 'dominance'] = score_dominance
    if len(data[data.isnull().any(axis=1)]) != 0:
        print("Null values included, cannot export to csv file")
        print(data)
        return
    data.to_csv(os.path.join(data_path, save_file))
    print('Successfully saved ', save_file)
    print('Data dimension: ', data.shape)


def load_data():
    base_path = './Functional_features/'
    partitions = ['train/', 'validation/', 'test/']
    # number of audio files
    num_train = 2702
    num_validation = 749
    num_test = 1503

    max_seq_len = 1937

    x_train = np.empty((num_train, max_seq_len, 0))
    x_val = np.empty((num_validation, max_seq_len, 0))
    x_test = np.empty((num_test, max_seq_len, 0))

    for i, partition in enumerate(partitions):
        if i == 0:
            x_train = np.concatenate( (x_train, load_features(path_features=base_path+partition, num_inst=num_train,
                                                              max_seq_len=max_seq_len) ), axis=2)
        elif i == 1:
            x_val = np.concatenate((x_val, load_features(path_features=base_path + partition, num_inst=num_validation,
                                                             max_seq_len=max_seq_len)), axis=2)
        else:
            x_test = np.concatenate((x_test, load_features(path_features=base_path + partition, num_inst=num_test,
                                                             max_seq_len=max_seq_len)), axis=2)

    print(x_train[0])
    print(x_val.shape)
    print(x_test.shape)


def main():
    # load_data()
    anno = get_annotations('train')
    print(anno)


if __name__ == '__main__':
    main()