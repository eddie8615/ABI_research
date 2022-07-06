import os
import json
import fnmatch
import numpy as np
import pandas as pd
from tqdm import tqdm
from read_csv import load_features
from extract_features_functional import get_annotations


folder_lld_features = ['../LLDs/train/',
                       '../LLDs/validation/',
                       '../LLDs/test/']
arouse_path = os.path.relpath('../MSP Data/Annotations/Arousal')
dominance_path = os.path.relpath('../MSP Data/Annotations/Dominance')
valence_path = os.path.relpath('../MSP Data/Annotations/Valence')

output_path = '../Data/'
# output_path = '../temp/'
label_output_path = '../Data/labels/'

sections = ['train', 'validation', 'test']
dimensions = ['Arousal', 'Valence', 'Dominance']


def main():
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(label_output_path):
        os.mkdir(label_output_path)
    train, validation, test = load_features()


def load_features():
    # Window size
    window_size = 4.0  # 4s
    # Hop length
    hop_size = 0.5  # 500ms
    # seq_length
    seq_len = 399
    # Minimum timp interval
    # According to Henrik Nordstorm (The time course of emotion recognition in speech and music,
    # The paper said that
    min_time = 0.25 # 250ms

    for i, folder in enumerate(folder_lld_features):
        # Initialise feature and label space for each partition
        features = []
        train = []
        validation = []
        test = []
        annotations = get_annotations(sections[i])

        arousal_label = []
        valence_label = []
        dominance_label = []
        filenames = []
        intervals = []

        # load segment.json
        segment_path = os.path.relpath('../MSP Data/Time Labels/segments.json')
        f = open(segment_path, 'r')
        timing_data = json.load(f)

        # Fetch all files
        files = fnmatch.filter(os.listdir(folder_lld_features[i]), '*.csv')
        files.sort()

        for file in tqdm(files):
            inst = file.split('.')[0]
            if 'MSP-PODCAST_0153' in inst or 'MSP-PODCAST_1188_0020' in inst:
                continue
            file_part = timing_data[inst]['Conversation_Part']
            start_offset = timing_data[inst]['Start_Time']

            # load annotations in Dataframe
            df_arousal, df_valence, df_dominance = load_annotation_df(annotations, file_part)

            x = pd.read_csv(folder_lld_features[i] + file)
            x = convert_timestamp(x)
            start = -hop_size
            end = start
            last_timestamp = max(x.iloc[:,1])
            while end < last_timestamp:

                # init emotion dimensions
                arousal = []
                valence = []
                dominance = []

                start += hop_size
                end = start + window_size
                if end > last_timestamp:
                    end = last_timestamp

                # Processing features
                features.append(get_features(start, end, x))

                # Processing labels
                for df in df_arousal:
                    df = df[(start + start_offset <= df['time']) & (df['time'] <= start_offset + end)]
                    if df.empty:
                        continue
                    arousal.append(df['arousal'].mean())
                for df in df_valence:
                    df = df[(start + start_offset <= df['time']) & (df['time'] <= start_offset + end)]
                    if df.empty:
                        continue
                    valence.append(df['valence'].mean())
                for df in df_dominance:
                    df = df[(start + start_offset <= df['time']) & (df['time'] <= start_offset + end)]
                    if df.empty:
                        continue
                    dominance.append(df['dominance'].mean())

                filenames.append(file.split('.')[0])
                intervals.append(str(start) +'-'+str(end))
                if len(arousal) == 0:
                    arousal = 0
                arousal = np.mean(arousal)
                arousal_label.append(arousal)
                if len(valence) == 0:
                    valence = 0
                valence = np.mean(valence)
                valence_label.append(valence)
                if len(dominance) == 0:
                    dominance = 0
                dominance = np.mean(dominance)
                dominance_label.append(dominance)

                # Debugging line
                # print("%s, %f-%f, Arousal: %f, Valence: %f, Dominance: %f" % (file.split('.')[0], start, end, arousal, valence, dominance))

        # Save the features
        if i == 0:
            train = features
            print('Train samples: %d' % (len(train)))
            save_features(train, output_path + sections[i] + '.txt')
        elif i == 1:
            validation = features
            print('Validation samples: %d' % (len(validation)))
            save_features(validation, output_path + sections[i] + '.txt')
        else:
            test = features
            print('Test samples: %d' % (len(test)))
            save_features(test, output_path + sections[i] + '.txt')

        # Save the labels
        di = {'Filename': filenames, 'Time': intervals, 'Arousal': arousal_label, 'Valence': valence_label, 'Dominance': dominance_label}
        df = pd.DataFrame(di)
        print('%s labels length: %d ' % (sections[i], len(df)))
        #
        # # Generate the labels
        save_labels(df, output_path + sections[i] + '_labels.txt')

    return train, validation, test


def load_annotation_df(annotations, file_part):
    a_annotations = annotations[0].get(file_part)
    v_annotations = annotations[1].get(file_part)
    d_annotations = annotations[2].get(file_part)
    df_arousal = []
    df_valence = []
    df_dominance = []
    for anno in a_annotations:
        pp = os.path.join(arouse_path, anno)
        df = pd.read_csv(pp, header=8, names=['time', 'arousal'])
        df_arousal.append(df)

    for anno in v_annotations:
        pp = os.path.join(valence_path, anno)
        df = pd.read_csv(pp, header=8, names=['time', 'valence'])
        df_valence.append(df)

    for anno in d_annotations:
        pp = os.path.join(dominance_path, anno)
        df = pd.read_csv(pp, header=8, names=['time', 'dominance'])
        if anno == 'MSP-Conversation_0047_2_001.csv':
            df = df.reset_index()
            df = df.drop(columns=['dominance'])
            df.columns = ['time', 'dominance']
        df_dominance.append(df)

    return df_arousal, df_valence, df_dominance

# def get_labels(start, end, annotations, file):
#     segment_path = os.path.relpath('../MSP Data/Time Labels/segments.json')
#     f = open(segment_path, 'r')
#     timing_data = json.load(f)
#     inst = file.split('.')[0]
#     file_part = timing_data[inst]['Conversation_Part']
#     a_annotations = annotations[0].get(file_part)
#     v_annotations = annotations[1].get(file_part)
#     d_annotations = annotations[2].get(file_part)
#     arousal = []
#     valence = []
#     dominance = []
#
#     for anno in a_annotations:
#         pp = os.path.join(arouse_path, anno)
#         df = pd.read_csv(pp, header=8, names=['time', 'arousal'])
#         df = df[(start <= df['time']) & (df['time'] <= end)]
#         if df.empty:
#             continue
#         arousal.append(df['arousal'].mean())
#     for anno in v_annotations:
#         pp = os.path.join(valence_path, anno)
#         df = pd.read_csv(pp, header=8, names=['time', 'valence'])
#         df = df[(start <= df['time']) & (df['time'] <= end)]
#         if df.empty:
#             continue
#         valence.append(df['valence'].mean())
#     for anno in d_annotations:
#         pp = os.path.join(dominance_path, anno)
#         df = pd.read_csv(pp, header=8, names=['time', 'dominance'])
#         if anno == 'MSP-Conversation_0047_2_001.csv':
#             df = df.reset_index()
#             df = df.drop(columns=['dominance'])
#             df.columns = ['time', 'dominance']
#         df = df[(start <= df['time']) & (df['time'] <= end)]
#         if df.empty:
#             continue
#         dominance.append(df['dominance'].mean())
#
#     return np.mean(arousal), np.mean(valence), np.mean(dominance)


def save_features(features, output_path):
    with open(output_path, 'w') as outfile:
        for slice_2d in features:
            np.savetxt(outfile, slice_2d)


def save_labels(labels, output_path):
    labels.to_csv(output_path)


def get_features(start, end, x, seq_len=399):
    window = np.zeros((seq_len, 25))
    sample = x[(start <= x['start']) & (end >= x['end'])]
    # print(sample)
    window[:sample.shape[0], :] = sample.iloc[:, 2:].to_numpy()
    return window


def convert_timestamp(x):
    # eGeMAPS configurations
    # time unit: sec
    # sample length: 20ms (0.02s)
    # hop_size: 10ms (0.01s)
    window_size = 0.02
    hop_size = 0.01
    for i in range(len(x)):
        x.iloc[i, 0] = i * hop_size
        x.iloc[i, 1] = x.iloc[i, 0] + window_size
    return x


if __name__ == '__main__':
    main()