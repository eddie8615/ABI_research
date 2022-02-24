import json
import os
from pydub import AudioSegment


def main():
    labels = file_labels()
    audio_segment(labels)


def audio_segment(labels):
    path = os.path.relpath('./MSP Data/Time Labels/segments.json')
    audio_path = os.path.relpath('./MSP Data/Audio')
    train_path = os.path.relpath('./Data/train')
    validation_path = os.path.relpath('./Data/validation')
    test_path = os.path.relpath('./Data/test')
    f = open(path, 'r')
    timing_data = json.load(f)
    sliced_files = []

    for key in timing_data:
        start_time = timing_data[key]['Start_Time'] * 1000
        end_time = timing_data[key]['End_Time'] * 1000
        save_path = ''

        if timing_data[key]['Conversation_Part'][0:21] in labels[0]:
            save_path = os.path.join(train_path, key+'.wav')
        elif timing_data[key]['Conversation_Part'][0:21] in labels[1]:
            save_path = os.path.join(validation_path, key+'.wav')
        else:
            save_path = os.path.join(test_path, key+'.wav')

        audio_file = timing_data[key]['Conversation_Part'][0:21] + '.wav'
        audio_file = os.path.join(audio_path, audio_file)
        file = AudioSegment.from_wav(audio_file)
        sliced = file[start_time:end_time]
        sliced.export(save_path, format="wav")
        sliced_files.append(sliced)
    print(len(sliced_files))


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
