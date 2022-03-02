import io
import os
from google.cloud import speech
import wave
from pydub import AudioSegment

audio_path = os.path.relpath('./Data/train/')
save_path = os.path.relpath('./transcripts/train')
bucket_name = 'msc_research'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/changhyun/workspace/ABI_research/config/google_services.json"


def main():
    # check_if_files_mono()
    # frame_rate_channel(sample_path)
    transcribing()


def transcribing():
    client = speech.SpeechClient()
    responses = []
    for file in os.listdir(audio_path):
        path = os.path.join(audio_path, file)
        with io.open(path, "rb") as audio_file:
            content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )
        try:
            responses.append(client.recognize(config=config, audio=audio))
        except:
            print(path)

    for response in responses:
        print("Transcript: {}".format(response.alternatives[0].transcript))


# The API does not support stereo audio files so that we need to check the audio files are mono
def check_if_files_mono():
    sample = AudioSegment.from_wav(audio_path)
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
