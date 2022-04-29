import os
import json
import pandas as pd
from tqdm import tqdm
import opensmile
import audiofile


# save path
sections = ['train', 'validation', 'test']
data_path = '../Data/'
output_folder = '../LLDs/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Initialise opensmile
# Feature set: eGeMAPS
# Feature level: LLD

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,

    num_workers=4
)


for section in sections:
    print('Starting extracting features of ' + section + ' files...')
    section_output_folder = output_folder + section + '/'
    if not os.path.exists(section_output_folder):
        os.mkdir(section_output_folder)

    path = data_path + section + '/'
    for audio in tqdm(os.listdir(path)):
        filename = path + audio
        instname = os.path.splitext(audio)[0]
        outfilename = section_output_folder + instname + '.csv'
        signal, sampling_rate = audiofile.read(filename, always_2d=True)
        data = smile.process_signal(signal, sampling_rate)
        # data.to_csv(outfilename)

    print('Ending extraction of ' + section + ' files...')