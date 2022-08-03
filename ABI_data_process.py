import os
import json
import pandas as pd
import fnmatch
from pyannote.audio import Pipeline
from tqdm import tqdm

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

data_path = './ABI_data/Audios/'
diary_path = data_path + 'diarization/'
if not os.path.exists(diary_path):
    os.mkdir(diary_path)
series = sorted(os.listdir(data_path))

min_duration = 0.5  # 500ms
for podcast in series:
    files = fnmatch.filter(os.listdir(data_path + podcast), '*.wav')
    files.sort()
    output_path = diary_path + podcast + '/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    count = 1
    for file in tqdm(files):
        inst = file.split('.')[0]
        diarization = pipeline(data_path + podcast + '/' + file)
        diary = dict()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            detail = dict()
            detail['episode'] = inst
            detail['start'] = turn.start
            detail['end'] = turn.end
            detail['speaker'] = speaker

            key = podcast + '_' + str(count).zfill(5)
            diary[key] = detail

            count += 1

        #             print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        with open(output_path + inst + '.json', "w") as outfile:
            json.dump(diary, outfile, indent=4)
