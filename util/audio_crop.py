import json
import os
from pydub import AudioSegment
import pandas as pd

train_path = os.path.relpath('../Data/train')
validation_path = os.path.relpath('../Data/validation')
test_path = os.path.relpath('../Data/test')


def main():
    labels = file_labels()

    audio_partition()
    audio_segment(labels)


def audio_segment(labels):
    path = os.path.relpath('../MSP Data/Time Labels/segments.json')
    part_path = os.path.relpath('../Data/partition/')
    f = open(path, 'r')
    timing_data = json.load(f)

    for key in timing_data:
        start_time = timing_data[key]['Start_Time'] * 1000
        end_time = timing_data[key]['End_Time'] * 1000

        if timing_data[key]['Conversation_Part'][0:21] in labels[0]:
            save_path = os.path.join(train_path, key+'.wav')
            # train = pd.concat([train, integrated], axis=0)
        elif timing_data[key]['Conversation_Part'][0:21] in labels[1]:
            save_path = os.path.join(validation_path, key+'.wav')
            # validation = pd.concat([validation, integrated], axis=0)
        else:
            save_path = os.path.join(test_path, key+'.wav')
            # test = pd.concat([test, integrated], axis=0)

        audio_file = timing_data[key]['Conversation_Part'] + '.wav'
        audio_file = os.path.join(part_path, audio_file)
        file = AudioSegment.from_wav(audio_file)
        sliced = file[start_time:end_time]
        sliced.export(save_path, format="wav")

    # train_csv = os.path.join(data_path, 'train.csv')
    # valid_csv = os.path.join(data_path, 'validation.csv')
    # test_csv = os.path.join(data_path, 'test.csv')
    #
    # train.to_csv(train_csv)
    # test.to_csv(test_csv)
    # validation.to_csv(valid_csv)


def audio_partition():
    path = os.path.relpath('../MSP Data/Time Labels/conversation_parts.txt')
    part_path = os.path.relpath('../Data/partition/')
    audio_path = os.path.relpath('../MSP Data/Audio')

    with open(path) as f:
        lines = f.readlines()

    prev = ''
    for line in lines:
        elem = line.strip().split(';')
        file_name = elem[0]+'.wav'
        audio = elem[0][0:21]

        if prev != audio:
            prev = audio
            offset = float(elem[1])
        save_path = os.path.join(part_path, file_name)
        start_time = (float(elem[1]) - offset) * 1000
        end_time = (float(elem[2]) - offset) * 1000
        audio_file = os.path.join(audio_path, elem[0][0:21]+'.wav')
        file = AudioSegment.from_wav(audio_file)
        sliced = file[start_time:end_time]
        sliced.export(save_path, format="wav")


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
    print('Number of training labels: ' + str(len(train_files)))
    print('Number of validation labels: ' + str(len(validation_files)))
    print('Number of testing labels: ' + str(len(test_files)))

    return [train_files, validation_files, test_files]


if __name__ == '__main__':
    main()
