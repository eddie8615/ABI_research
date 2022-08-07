import os
import io
import json
import audiofile
import opensmile
import wave
import fnmatch
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
from google.cloud import storage
from google.cloud import speech


data_path = '../ABI_data/'
audio_path = data_path + 'Audios/'
diary_path = data_path + 'diarization/'
temp_path = data_path + 'temp/'

# This path will be followed by series name
output_path_base = data_path + 'Transcripts/'
lld_path = data_path + 'LLDs/'

failed_path = data_path + 'transcribing_failed.txt'
bucket_name = 'msc_research_kings'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/changhyun/workspace/ABI_research/config/config3.json"


def main():
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,

        num_workers=4
    )
    with open(data_path+'selected.txt', 'r') as f:
        lines = f.readlines()
        series = []
        for line in lines:
            series.append(line.strip('\n'))

    conf_dict = dict()

    for s in series:
        s_diaries = os.listdir(diary_path + s)
        episodes = fnmatch.filter(s_diaries, '*_cleaned.json')
        episodes.sort()

        confidences = []
        for episode in tqdm(episodes):
            f = open(diary_path + s + '/' + episode)
            data = json.load(f)

            inst = episode.split('.')[0].split('_')[0]
            audio_file_path = audio_path + s + '/' + inst + '.wav'
            file = AudioSegment.from_wav(audio_file_path)
            output_lld_path = lld_path + s + '/'
            if not os.path.exists(output_lld_path):
                os.mkdir(output_lld_path)

            print('Current episode: %s' % (inst))

            for key in data:
                segment = data[key]
                start = segment['start'] * 1000
                end = segment['end'] * 1000
                sliced = file[start:end]
                temp_file_path = temp_path + key + '.wav'
                sliced.export(temp_file_path, format="wav")
                if end - start < 60.0:
                    transcript, conf = short_transcribe(temp_file_path)
                else:
                    transcript, conf = long_transcribe(temp_file_path)

                if conf == 0:
                    print("Failed to transcribe:", key)
                    fi = open(failed_path, "a")
                    fi.write(key + '\n')
                    fi.close()

                confidences.append(conf)
                new_path = os.path.join(output_path_base + s, key + '.txt')
                if not os.path.exists(output_path_base + s):
                    os.mkdir(output_path_base + s)
                write_transcripts(new_path, transcript)
                print(key, '.txt has been created')

                extracted = smile.process_file(temp_file_path)
                extracted.to_csv(output_lld_path + key + '.csv')

                os.remove(temp_file_path)

        conf_dict[inst] = confidences



def short_transcribe(audio_file_name):
    client = speech.SpeechClient()
    confidences = []
    transcript=''
    frame_rate, channels = frame_rate_channel(audio_file_name)
    value = False
    if channels > 1:
        value = True
    with io.open(audio_file_name, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=frame_rate,
        audio_channel_count=channels,
        enable_automatic_punctuation=True,
        enable_separate_recognition_per_channel=value,
        language_code="en-US",
    )
    try:
        response = client.recognize(config=config, audio=audio)
    except:
        print(audio_file_name)
    for result in response.results:
        transcript += result.alternatives[0].transcript
        confidences.append(result.alternatives[0].confidence)

    return transcript, mean(confidences)


def long_transcribe(audio_file_name):
    #     file_name = filepath + audio_file_name

    # The name of the audio file to transcribe

    frame_rate, channels = frame_rate_channel(audio_file_name)

    #     source_file_name = filepath + audio_file_name
    destination_blob_name = audio_file_name

    upload_blob(bucket_name, audio_file_name, destination_blob_name)

    gcs_uri = 'gs://' + bucket_name + '/' + audio_file_name
    transcript = ''

    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)
    value = False
    if channels > 1:
        value = True

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=frame_rate,
        audio_channel_count=channels,
        enable_automatic_punctuation=True,
        enable_separate_recognition_per_channel=value,
        language_code='en-US')

    # Detects speech in the audio file
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=10000)
    confidence = []

    for result in response.results:
        transcript += result.alternatives[0].transcript
        confidence.append(result.alternatives[0].confidence)

    delete_blob(bucket_name, destination_blob_name)
    return transcript, mean(confidence)



def frame_rate_channel(audio_file):
    with wave.open(audio_file, "r") as wf:
        frame_rate = wf.getframerate()
        channels = wf.getnchannels()
        return frame_rate, channels


def frame_rate_channel_freq(audio_path):
    frame_rates = {}
    channels = {}
    for file in os.listdir(audio_path):
        path = os.path.join(audio_path, file)
        with wave.open(path, "r") as wf:
            frame_rate = wf.getframerate()
            channel = wf.getnchannels()
            freq = frame_rates.get(frame_rate, "None")
            if freq == "None":
                frame_rates[frame_rate] = 1
            else:
                frame_rates[frame_rate] += 1
            freq = channels.get(channel, "None")
            if freq == "None":
                channels[channel] = 1
            else:
                channels[channel] += 1
    return frame_rates, channels


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()


def mean(li):
    if len(li) == 0:
        return 0
    return sum(li) / len(li)


def byte_to_mb(size):
    return size / 1024 / 1024


def write_transcripts(transcript_filename,transcript):
    f= open(transcript_filename,"w+")
    f.write(transcript)
    f.close()


if __name__ == '__main__':
    main()
