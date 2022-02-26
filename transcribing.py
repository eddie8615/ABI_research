import io
import os
from google.cloud import speech
import wave
from pydub import AudioSegment

sample_path = os.path.relpath('./Data/train/MSP-PODCAST_0021_0003.wav')
bucket_name = 'msc_research'
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/changhyun/workspace/ABI_research/config/google_services.json"

def main():
    # check_if_files_mono()
    # frame_rate_channel(sample_path)
    transcribing()

def transcribing():
    client = speech.SpeechClient()
    with io.open(sample_path, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)
    for result in response.results:
        print("Transcript: {}".format(result.alternatives[0].transcript))


# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     """Uploads a file to the bucket."""
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
#     blob.upload_from_filename(source_file_name)
#
#
# def delete_blob(bucket_name, blob_name):
#     """Deletes a blob from the bucket."""
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(blob_name)
#
#     blob.delete()


# The API does not support stereo audio files so that we need to check the audio files are mono
def check_if_files_mono():
    sample = AudioSegment.from_wav(sample_path)
    print(sample.channels)


def frame_rate_channel(audio_file):
    with wave.open(audio_file, "r") as wf:
        frame_rate = wf.getframerate()
        channels = wf.getnchannels()
        print(frame_rate)
        print(channels)
        return frame_rate, channels


if __name__ == '__main__':
    main()
