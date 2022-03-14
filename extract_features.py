import os
import json
import pandas as pd
from audio_crop import file_labels
import opensmile


# save paths
arouse_path = os.path.relpath('./MSP Data/Annotations/Annotations/Arousal')
dominance_path = os.path.relpath('./MSP Data/Annotations/Annotations/Dominance')
valence_path = os.path.relpath('./MSP Data/Annotations/Annotations/Valence')
train_path = os.path.relpath('./Data/train')
validation_path = os.path.relpath('./Data/validation')
test_path = os.path.relpath('./Data/test')
data_path = os.path.relpath('./Data')


def main():

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_workers=4
    )

    # annotations = get_annotations('train')
    # extract_features(smile, annotations, 'train')
    # annotations = get_annotations('validation')
    # extract_features(smile, annotations, 'validation')
    annotations = get_annotations('test')
    extract_features(smile, annotations, 'test')


def extract_features(smile, annotations, phase=None):
    file_paths = []
    offset_index = 0
    if phase is None:
        return
    if phase == 'train':
        audio_file_path = train_path
        save_file = 'train.csv'
        offset_index = 11
    elif phase == 'test':
        audio_file_path = test_path
        save_file = 'test.csv'
        offset_index = 10
    elif phase == 'validation':
        audio_file_path = validation_path
        save_file = 'validation.csv'
        offset_index = 16
    else:
        print("Unknown phase called, terminated")

    for file in os.listdir(audio_file_path):
        file_paths.append(os.path.join(audio_file_path, file))
    # Extract acoustic features using opensmile
    data = smile.process_files(file_paths)

#     Indices in extracted dataframe were multi-index format.
#     Make them into single index format and the indices are the name of audio files
    data = data.reset_index(level=[1, 2])
#     drop nan
    data = data.dropna()
    # Replace indices from file path to file name
    indices = []
    for index in data.index:
        indices.append(index[offset_index:offset_index+21])

    data['podcast'] = indices
    data = data.set_index('podcast')
    segment_path = os.path.relpath('./MSP Data/Time Labels/segments.json')
    f = open(segment_path, 'r')
    timing_data = json.load(f)
    for key in timing_data:
        conv_part = timing_data[key]['Conversation_Part']
        if key not in data.index:
            continue
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


def get_annotations(data_phase='train'):
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
        print('Invalid argument')
        return

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


def mean(list):
    return sum(list) / len(list)


if __name__ == '__main__':
    main()
