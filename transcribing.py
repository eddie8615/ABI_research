import os
import io
from google.cloud import speech
import wave
from pydub import AudioSegment
from tqdm import tqdm
from google.cloud import storage

train_path = os.path.relpath('./Data/train/')
test_path = os.path.relpath('./Data/test')
validation_path = os.path.relpath('./Data/validation')
train_write_path = os.path.relpath('./transcripts/train')
test_write_path = os.path.relpath('./transcripts/test')
validation_write_path = os.path.relpath('./transcripts/validation')
failed_path = os.path.relpath('./failed.txt')
save_path = os.path.relpath('.')
bucket_name = 'msc_research'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/changhyun/workspace/ABI_research/config/config.json"


def main():
    train_long_files, train_short_files = find_long_audios(train_path)
    test_long_files, test_short_files = find_long_audios(test_path)
    validation_long_files, validation_short_files = find_long_audios(validation_path)

    print('Training files transcribing...')
    transcribe(train_path, train_write_path, train_long_files, train_short_files)
    # print('Testing files transcribing...')
    # transcribe(test_path, test_write_path, test_long_files, test_short_files)
    # print('Validation files transcribing...')
    # transcribe(validation_path, validation_write_path, validation_long_files, validation_short_files)


def transcribe(path, write_path, long_files, short_files):
    long_confidences = []
    short_confidences = []
    for file in long_files:
        file_name = os.path.join(path, file)
        transcript, confidence = long_transcribe(file_name)
        if confidence == 0:
            print("Failed to transcribe:", file_name)
            fi = open(failed_path, "a")
            fi.write(file_name+'\n')
            fi.close()

        long_confidences.append(confidence)
        new_path = os.path.join(write_path, file[0:21] + '.txt')
        write_transcripts(new_path, transcript)
        print(file[0:21], '.txt has been created')

    print('End of long files')
    for file in short_files:
        file_name = os.path.join(path, file)
        transcript, confidence = short_transcribe(file_name)
        if confidence == 0:
            print("Failed to transcribe:", file_name)
            fi = open(failed_path, "a")
            fi.write(file_name + '\n')
            fi.close()

        short_confidences.append(confidence)
        new_path = os.path.join(write_path, file[0:21] + '.txt')
        write_transcripts(new_path, transcript)
        print(file[0:21], '.txt has been created')

    print('Average confidences for long files:', mean(long_confidences))
    print('Average confidences for short files:', mean(short_confidences))


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

# The API does not support stereo audio files so that we need to check the audio files are mono
# def check_if_files_mono():
#     sample = AudioSegment.from_wav(audio_path)
#     print(sample.channels)


def write_transcripts(transcript_filename,transcript):
    f= open(transcript_filename,"w+")
    f.write(transcript)
    f.close()


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


# limit 60sec & 10MB
def find_long_audios(path):
    long_files = []
    short_files = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        size = byte_to_mb(os.path.getsize(file_path))
        if size > 10:
            long_files.append(file)
            continue

        with wave.open(file_path, "r") as wf:
            frame_rate = wf.getframerate()
            channel = wf.getnchannels()
            n_frames = wf.getnframes()
            duration = n_frames / float(frame_rate)
            if duration > 60:
                long_files.append(file)
            else:
                short_files.append(file)
    return long_files, short_files


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


if __name__ == '__main__':
    main()
