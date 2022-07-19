import os
import json
import pandas as pd
import numpy as np
import fnmatch
from tqdm import tqdm
# import opensmile


# save paths
arouse_path = os.path.relpath('../MSP Data/Annotations/Arousal')
dominance_path = os.path.relpath('../MSP Data/Annotations/Dominance')
valence_path = os.path.relpath('../MSP Data/Annotations/Valence')
train_path = os.path.relpath('../Data/train')
validation_path = os.path.relpath('../Data/validation')
test_path = os.path.relpath('../Data/test')
data_path = os.path.relpath('../Data')

window_size = 4.0
fps = 100
n_features = 25

window_size_half = int(window_size * fps / 2)
hop_size = 0.5
seq_len = 400
max_time = 193.62
max_seq_len = int(max_time / hop_size) + 1
min_time = 1.0
folder_lld_features = ['../LLDs/train/',
                   '../LLDs/validation/',
                   '../LLDs/test/']
sections = ['train', 'validation', 'test']
dimensions = ['Arousal', 'Valence', 'Dominance']
output_path = '../Functional_features/'

def main():
    # output_path = '../Data/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    extract_features()

    
def get_annotations(data_phase=None):
    train_files, validation_files, test_files = file_labels()

    annotations = []
    emotion_paths = []
    emotion_paths.append(arouse_path)
    emotion_paths.append(valence_path)
    emotion_paths.append(dominance_path)
    if data_phase == 'train':
        data_file = train_files
    elif data_phase == 'validation':
        data_file = validation_files
    elif data_phase == 'test':
        data_file = test_files
    else:
        data_file = train_files + validation_files + test_files

    for path in emotion_paths:
        annotation = {}
        li = os.listdir(path)
        list1 = []
        for file in li:
            if file[0:21] in data_file:
                list1.append(file)
        list1.sort()
        for elem in list1:
            if elem[0:23] in annotation.keys():
                annotation.get(elem[0:23]).append(elem)
            else:
                annotation[elem[0:23]] = [elem]
        annotations.append(annotation)
    return annotations

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

def save_features(features, output_path):
    with open(output_path, 'w') as outfile:
        for slice_2d in features:
            np.savetxt(outfile, slice_2d)


def save_labels(labels, output_path):
    labels.to_csv(output_path)
    
    
def extract_features():
    segment_path = os.path.relpath('../MSP Data/Time Labels/segments.json')
    f = open(segment_path, 'r')
    timing_data = json.load(f)

    max_time = 0
    for key in timing_data:
        start = timing_data[key]['Start_Time']
        end = timing_data[key]['End_Time']
        max_time = max(max_time, end-start)
    
    for i, folder in enumerate(folder_lld_features):

        arousal_label = []
        valence_label = []
        dominance_label = []
        filenames = []
        intervals = []

        features = []
        labels = []
        train = []
        validation = []
        test = []

        annotations = get_annotations(sections[i])
        files = fnmatch.filter(os.listdir(folder_lld_features[i]), '*.csv')
        files.sort()

        for file in tqdm(files):
            inst = file.split('.')[0]        
            if 'MSP-PODCAST_0153' in inst or 'MSP-PODCAST_1188_0020' in inst:
                continue
            file_part = timing_data[inst]['Conversation_Part']
            start_offset = timing_data[inst]['Start_Time']
            end_time = timing_data[inst]['End_Time']
            duration = end_time - start_offset
            if duration < min_time:
                continue
            # load annotations in Dataframe
            df_arousal, df_valence, df_dominance = load_annotation_df(annotations, file_part)

            x_func = np.zeros((max_seq_len, n_features * 2))
            labels_seq = np.zeros((max_seq_len, 3))
            x = pd.read_csv(folder_lld_features[i] + file)
            x = convert_timestamp(x)

            time_stampes_new = np.empty((max_seq_len, 1))
            for t in range(0, max_seq_len):
                arousal = []
                valence = []
                dominance = []

                t_orig = int(t * fps * hop_size)
                min_orig = max(0, t_orig-window_size_half)
                max_orig = min(x.shape[0], t_orig+window_size_half)
                start = min_orig / fps
                end = max_orig / fps

                if min_orig<max_orig and t_orig<=x.shape[0]:

                    x_func[t, :n_features] = np.mean(x.iloc[min_orig:max_orig, 2:], axis=0)
                    x_func[t, n_features:] = np.std(x.iloc[min_orig:max_orig, 2:], axis=0)
    #                 x_lld = x.iloc[min_orig:max_orig,2:]
    #                 time_stamps_new[:len(x_lld),:] = x_lld
    #                 features.append(time_stamps_new)
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
                    labels_seq[t, 0] = arousal
    #                 arousal_label.append(arousal)
                    if len(valence) == 0:
                        valence = 0
                    valence = np.mean(valence)
                    labels_seq[t, 1] = valence
    #                 valence_label.append(valence)
                    if len(dominance) == 0:
                        dominance = 0
                    dominance = np.mean(dominance)
                    labels_seq[t, 2] = dominance
    #                 dominance_label.append(dominance)

                else:
                    x_func = x_func[:t,:]                
                    x_func = np.concatenate((x_func, np.zeros((max_seq_len - x_func.shape[0], x_func.shape[1]))))
                    labels_seq = np.concatenate((labels_seq, np.zeros((max_seq_len- labels_seq.shape[0], 3))))
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
    #     di = {'Filename': filenames, 'Time': intervals, 'Arousal': arousal_label, 'Valence': valence_label, 'Dominance': dominance_label}
    #     df = pd.DataFrame(di)
    #     print('%s labels length: %d ' % (sections[i], len(df)))
        #
        # # Generate the labels
        save_features(labels, output_path + sections[i] + '_labels.txt')


def file_labels():
    filepath = os.path.relpath('../MSP Data/partitions.txt')
    with open(filepath, 'r') as f:
        lines = f.readlines()

    train_files = []
    validation_files = []
    test_files = []

    for line in lines:
        text = line.strip().split(';')
        if text[1] == 'Train':
            train_files.append(text[0])
        if text[1] == 'Validation':
            validation_files.append(text[0])
        if text[1] == 'Test':
            test_files.append(text[0])
    f.close()
    # print('Number of training labels: ' + str(len(train_files)))
    # print('Number of validation labels: ' + str(len(validation_files)))
    # print('Number of testing labels: ' + str(len(test_files)))

    return [train_files, validation_files, test_files]

if __name__ == '__main__':
    main()
