import os
import sys
import fnmatch
import numpy as np
import pandas as pd
from tqdm import tqdm
from read_csv import load_features
from wrtie_csv import save_features


folder_lld_features = ['../LLDs/train/',
                       '../LLDs/validation/',
                       '../LLDs/test/']

output_path = '../Functional_features/'
label_output_path = '../Functional_features/labels/'

if not os.path.exists(output_path):
    os.mkdir(output_path)
if not os.path.exists(label_output_path):
    os.mkdir(label_output_path)

sections = ['train', 'validation', 'test']
dimensions = ['Arousal', 'Valence', 'Dominance']

# Window size
window_size = 4.0  # 4s
# Hop length
hop_size = 0.5  # 500ms

for i, folder in enumerate(folder_lld_features):

    # Fetch all files
    files = fnmatch.filter(os.listdir(folder_lld_features[i]), '*.csv')
    files.sort()

    section_output_path = output_path + sections[i] + '/'
    section_label_output = label_output_path + sections[i] + '/'
    if not os.path.exists(section_output_path):
        os.mkdir(section_output_path)
    if not os.path.exists(section_label_output):
        os.mkdir(section_label_output)

    for file in tqdm(files):
        x = pd.read_csv(folder_lld_features[i] + file)

        # eGeMAPS configurations
        # time unit: sec
        # sample length: 20ms (0.02s)
        # hop_size: 10ms (0.01s)
        window_size = 0.02
        hop_size = 0.01

        for i in range(len(x)):
            x.iloc[i, 0] = i * hop_size
            x.iloc[i, 1] = x.iloc[i, 0] + window_size

#       TODO: Finished converting timestamp, then extract features in 4s window chunk and save at the save path
