import json
import os
from pydub import AudioSegment
import pandas as pd

arouse_path = os.path.relpath('./MSP Data/Annotations/Annotations/Arousal/MSP-Conversation_0021_1_001.csv')
dominance_path = os.path.relpath('./MSP Data/Annotations/Annotations/Dominance')
valence_path = os.path.relpath('./MSP Data/Annotations/Annotations/Valence')
train_path = os.path.relpath('./Data/train')
validation_path = os.path.relpath('./Data/validation')
test_path = os.path.relpath('./Data/test')


def main():
    labels = file_labels()
    data = pd.read_csv(arouse_path, header=8, names=['time', 'arousal'])
    print(data.head())
    # audio_partition()
    # audio_segment(labels)


def audio_segment(labels):
    path = os.path.relpath('./MSP Data/Time Labels/segments.json')
    part_path = os.path.relpath('./Data/partition/')

    f = open(path, 'r')
    timing_data = json.load(f)

    for key in timing_data:
        start_time = timing_data[key]['Start_Time'] * 1000
        end_time = timing_data[key]['End_Time'] * 1000

        if timing_data[key]['Conversation_Part'][0:21] in labels[0]:
            save_path = os.path.join(train_path, key+'.wav')
            save_y = os.path.join(train_path, key+'.csv')
        elif timing_data[key]['Conversation_Part'][0:21] in labels[1]:
            save_path = os.path.join(validation_path, key+'.wav')
            save_y = os.path.join(train_path, key + '.csv')
        else:
            save_path = os.path.join(test_path, key+'.wav')
            save_y = os.path.join(train_path, key + '.csv')

        audio_file = timing_data[key]['Conversation_Part'] + '.wav'
        audio_file = os.path.join(part_path, audio_file)
        file = AudioSegment.from_wav(audio_file)
        sliced = file[start_time:end_time]
        sliced.export(save_path, format="wav")
        arouse = pd.read_csv()


def audio_partition():
    path = os.path.relpath('./MSP Data/Time Labels/conversation_parts.txt')
    part_path = os.path.relpath('./Data/partition/')
    audio_path = os.path.relpath('./MSP Data/Audio')

    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        elem = line.strip().split(';')
        file_name = elem[0]+'.wav'
        save_path = os.path.join(part_path, file_name)
        start_time = float(elem[1]) * 1000
        end_time = float(elem[2]) * 1000
        audio_file = os.path.join(audio_path, elem[0][0:21]+'.wav')
        file = AudioSegment.from_wav(audio_file)
        sliced = file[start_time:end_time]
        sliced.export(save_path, format="wav")


def file_labels():
    filepath = os.path.relpath('./MSP Data/partitions.txt')
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
    print('Number of training labels: ' + str(len(train_files)))
    print('Number of validation labels: ' + str(len(validation_files)))
    print('Number of testing labels: ' + str(len(test_files)))

    return [train_files, validation_files, test_files]


if __name__ == '__main__':
    main()
