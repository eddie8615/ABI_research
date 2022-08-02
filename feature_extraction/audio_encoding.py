import os
import json
import audiofile
import pandas as pd
# from pyannote.audio import Pipeline
from tqdm import tqdm

data_path = '../ABI_data/Audios/'

series = os.listdir(data_path)
series.sort()

failed = []

for s in tqdm(series):
    path = data_path + s + '/'
    audios = os.listdir(path)
    audios.sort()
    print('Current series: %s' % s)
    for audio in audios:
        # Encode all audio files from mp3 to wav
        try:
            signal, rate = audiofile.read(path + audio)
        except:
            failed.append(path+audio)
            continue

        inst = audio.split('.')[0]
        audiofile.write(path+inst+'.wav', signal, rate)

with open(r'../encoding_failed.txt', 'w') as f:
    for item in failed:
        f.write("%s\n" % item)
    f.close()
    print("write done")