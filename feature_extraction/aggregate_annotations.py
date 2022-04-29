import os
import json
import pandas as pd
import numpy as np
from numpy import mean

arouse_path = os.path.relpath('../MSP Data/Annotations/Arousal')
dominance_path = os.path.relpath('../MSP Data/Annotations/Dominance')
valence_path = os.path.relpath('../MSP Data/Annotations/Valence')
train_path = os.path.relpath('../Data/train')
validation_path = os.path.relpath('../Data/validation')
test_path = os.path.relpath('../Data/test')
data_path = os.path.relpath('../Data')

output_path = '../labels/'
dimensions = ['Arousal', 'Valence', 'Dominance']


def main():
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    annoations = get_annotations()


def aggregate_labels(annotations):
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


def get_annotations():
    annotations = []
    emotion_paths = []
    emotion_paths.append(arouse_path)
    emotion_paths.append(valence_path)
    emotion_paths.append(dominance_path)

    for path in emotion_paths:
        annotation = {}
        li = os.listdir(path)
        list1 = []
        for file in li:
            list1.append(file)

        list1.sort()
        for elem in list1:
            if elem[0:23] in annotation.keys():
                annotation.get(elem[0:23]).append(elem)
            else:
                annotation[elem[0:23]] = [elem]
        annotations.append(annotation)
    return annotations


if __name__ == '__main__':
    main()