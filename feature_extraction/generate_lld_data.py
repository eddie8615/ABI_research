import os
import json
import fnmatch
import numpy as np
import pandas as pd
import contextlib
import wave
import argparse
from tqdm import tqdm
from read_csv import load_features
from extract_features_functional import get_annotations

arouse_path = os.path.relpath('../MSP Data/Annotations/Arousal')
dominance_path = os.path.relpath('../MSP Data/Annotations/Dominance')
valence_path = os.path.relpath('../MSP Data/Annotations/Valence')

sections = ['train', 'validation', 'test']
dimensions = ['Arousal', 'Valence', 'Dominance']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence', type=str, required=True)
    args = parser.parse_args()
    args = args.sentence

    if args == 'y' or args == 'yes':
        sentence_level = True
        output_path = '../LLDs_podcast/'

    elif args == 'n' or args == 'no':
        sentence_level = False
        output_path = '../LLDs_conversation'
    else:
        print('Invalid argument')
        return

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    generate_features(output_path, sentence_level=sentence_level)


def generate_features(output_path, sentence_level=True):
    # Feature configurations
    # Window size
    window_size = 4.0  # 4s
    # Hop length
    hop_size = 0.5  # 500ms
    # seq_length
    seq_len = 400

    # Minimum timp interval
    min_time = 1.0 # 1s
    # Frame per second (openSmile produces llds with 0.01s hop_size)
    fps = 100
    # Number of features
    n_features = 25

    window_size_half = int(window_size * fps / 2)
    # load segment.json
    segment_path = os.path.relpath('../MSP Data/Time Labels/segments.json')
    f = open(segment_path, 'r')
    timing_data = json.load(f)

    if sentence_level:
        max_duration = find_maximum_length(timing_data)
        folder_lld_features = ['../LLDs_podcast/train/',
                               '../LLDs_podcast/validation/',
                               '../LLDs_podcast/test/']
    else:
        max_duration = find_longest_podcast()
        folder_lld_features = ['../LLDs_conversation/train/',
                               '../LLDs_conversation/validation/',
                               '../LLDs_conversation/test/']
    # maximum sequence length
    max_seq_len = int(max_duration / hop_size) + 1

    for i, folder in enumerate(folder_lld_features):
        # Initialise feature and label space for each partition
        features = []
        annotations = get_annotations(sections[i])

        # reaction lag compensation according to MSP-conversation paper
        arousal_lag = 2.8
        valence_lag = 4.08
        dominance_lag = 2.8

        arousal_label = []
        valence_label = []
        dominance_label = []
        labels = []
        filenames = []
        intervals = []

        # Fetch all files
        files = fnmatch.filter(os.listdir(folder_lld_features[i]), '*.csv')
        files.sort()

        for file in tqdm(files):
            inst = file.split('.')[0]
            if 'MSP-PODCAST_0153' in inst or 'MSP-PODCAST_1188_0020' in inst:
                continue

            if sentence_level:
                file_part = timing_data[inst]['Conversation_Part']
                start_offset = timing_data[inst]['Start_Time']
                end_time = timing_data[inst]['End_Time']
                duration = end_time - start_offset
                if duration < min_time:
                    continue
                # load annotations in Dataframe
                df_arousal, df_valence, df_dominance = load_annotation_df(annotations, file_part)
            else:
                start_offset = 0
                df_arousal, df_valence, df_dominance = load_annotation_df(annotations, inst)

            labels_seq = np.zeros((max_seq_len, 3))
            x_func = np.zeros((max_seq_len, n_features * 2))
            x = pd.read_csv(folder_lld_features[i] + file)
            x = convert_timestamp(x)
            for t in range(0, max_seq_len):
                arousal = []
                valence = []
                dominance = []

                t_orig = int(t * fps * hop_size)
                min_orig = max(0, t_orig - window_size_half)
                max_orig = min(x.shape[0], t_orig + window_size_half)
                start = min_orig / fps
                end = max_orig / fps

                if min_orig < max_orig and t_orig <= x.shape[0]:

                    x_func[t, :n_features] = np.mean(x.iloc[min_orig:max_orig, 2:], axis=0)
                    x_func[t, n_features:] = np.std(x.iloc[min_orig:max_orig, 2:], axis=0)

                    for df in df_arousal:
                        df = df[(start + start_offset + arousal_lag <= df['time']) & (
                                    df['time'] <= end + start_offset + arousal_lag)]
                        if df.empty:
                            continue
                        arousal.append(df['arousal'].mean())
                    for df in df_valence:
                        df = df[(start + start_offset + valence_lag <= df['time']) & (
                                    df['time'] <= end + start_offset + valence_lag)]
                        if df.empty:
                            continue
                        valence.append(df['valence'].mean())
                    for df in df_dominance:
                        df = df[(start + start_offset + dominance_lag <= df['time']) & (
                                    df['time'] <= end + start_offset + dominance_lag)]
                        if df.empty:
                            continue
                        dominance.append(df['dominance'].mean())

                    filenames.append(file.split('.')[0])
                    intervals.append(str(start) + '-' + str(end))
                    if len(arousal) == 0:
                        arousal = 0
                    arousal = np.mean(arousal)
                    labels_seq[t, 0] = arousal
                    #  arousal_label.append(arousal)
                    if len(valence) == 0:
                        valence = 0
                    valence = np.mean(valence)
                    labels_seq[t, 1] = valence
                    # valence_label.append(valence)
                    if len(dominance) == 0:
                        dominance = 0
                    dominance = np.mean(dominance)
                    labels_seq[t, 2] = dominance
                #    dominance_label.append(dominance)

                else:
                    x_func = x_func[:t,:]
                    x_func = np.concatenate((x_func, np.zeros((max_seq_len - x_func.shape[0], x_func.shape[1]))))
                    labels_seq = np.concatenate((labels_seq, np.zeros((max_seq_len - labels_seq.shape[0], 3))))
                    break

            features.append(x_func)
            labels.append(labels_seq)
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

    # # Generate the labels
    save_features(labels, output_path + sections[i] + '_labels_lag_compensated.txt')


def find_maximum_length(timing_data):

    max_duration = 0
    for key in timing_data:
        start = timing_data[key]['Start_Time']
        end = timing_data[key]['End_Time']

        duration = end - start
        max_duration = max(duration, max_duration)

    return max_duration


def find_longest_podcast():
    audio_path = '../Data/partition/'
    files = fnmatch.filter(os.listdir(audio_path), '*.wav')
    files.sort()
    max_duration = 0
    for file in files:
        with contextlib.closing(wave.open(audio_path+file,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            max_duration = max(max_duration, duration)
    return max_duration


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


def save_features(features, output_path):
    with open(output_path, 'w') as outfile:
        for slice_2d in features:
            np.savetxt(outfile, slice_2d)


def save_labels(labels, output_path):
    labels.to_csv(output_path)


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