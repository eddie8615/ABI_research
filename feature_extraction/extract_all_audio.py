import os
import fnmatch
import json
import pandas as pd
from tqdm import tqdm
import opensmile
import audiofile
from audio_crop import file_labels

data_path = '../Data/partition/'
output_path = '../LLD_temp/'
sections = ['train/', 'validation/', 'test/']

'''
This file is to extract acoustic feature through openSmile toolkit.

Extract low-level descriptors from the whole chunk audio files.
'''

# mkdir if there is no directory
if not os.path.exists(output_path):
    os.mkdir(output_path)
    for section in sections:
        if not os.path.exists(output_path+section):
            os.mkdir(output_path+section)

# load files
train_files, val_files, test_files = file_labels()

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,

    num_workers=4
)

files = fnmatch.filter(os.listdir(data_path), '*.wav')
files.sort()

# Instance offset 21 (MSP-Conversation_xxxx_xxxx)
inst_offset = 21
for file in tqdm(files):
    inst = file.split('.')[0]

    offset = file[:inst_offset]
    if offset in train_files:
        out_path = output_path + sections[0]
    elif offset in val_files:
        out_path = output_path + sections[1]
    else:
        out_path = output_path + sections[2]

    signal, sampling_rate = audiofile.read(data_path+file, always_2d=True)
    data = smile.process_signal(signal, sampling_rate)
    data.to_csv(out_path + inst + '.csv')
    print('Finished %s' % (inst + '.csv'))