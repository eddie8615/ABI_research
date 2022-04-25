import os
import sys
import fnmatch
import numpy as np
from tqdm import tqdm
from read_csv import load_features
from wrtie_csv import save_features


folder_lld_features = ['../LLDs/train/',
                       '../LLDs/validation/',
                       '../LLDs/test/']

output_path = '../Functional_features/'

if not os.path.exists(output_path):
    os.mkdir(output_path)

sections = ['train', 'validation', 'test']

# Window size
window_size = 4.0

fps = 100
max_seq_len = 1768
hop_size = 0.1

for i, folder in enumerate(folder_lld_features):

    # Fetch all files
    files = fnmatch.filter(os.listdir(folder_lld_features[i]), '*.csv')
    files.sort()

    section_output_path = output_path + sections[i] + '/'
    if not os.path.exists(section_output_path):
        os.mkdir(section_output_path)

    for file in tqdm(files):
        # X = load_features(os.path.abspath(folder_lld_features[i] + file), skip_header=True, skip_instname=True, delim=',')
        X = np.genfromtxt(folder_lld_features[i] + file, skip_header=1, delimiter=',')

        if X.ndim < 2:
            continue
        num_llds = X.shape[1] - 2
        X_func = np.zeros((max_seq_len, num_llds*2))
        window_size_half = int(window_size * fps / 2)

        time_stamps_new = np.empty((max_seq_len, 1))
        for t in range(0, max_seq_len):
            t_orig = int(t * fps * hop_size)
            min_orig = max(0, t_orig - window_size_half)
            max_orig = min(X.shape[0], t_orig + window_size_half + 1)
            if min_orig < max_orig and t_orig <= X.shape[0]:  # X can be smaller, do not consider
                time_stamps_new[t] = t * hop_size
                X_func[t, :num_llds] = np.mean(X[min_orig:max_orig, 2:], axis=0)  # skip time stamp
                X_func[t, num_llds:] = np.std(X[min_orig:max_orig, 2:], axis=0)  # skip time stamp
            else:
                time_stamps_new = time_stamps_new[:t, :]
                X_func = X_func[:t, :]
                break
        X_func = np.concatenate((time_stamps_new, X_func), axis=1)

        save_features(section_output_path + file, X_func, append=False, instname=file[:-4], header='', delim=';',precision=6, first_time=True)
