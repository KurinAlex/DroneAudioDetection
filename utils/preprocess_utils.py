import glob
import os

from moviepy import AudioFileClip, VideoFileClip
from pydub import AudioSegment


def extract_sound(source_dir, destination_dir, filename_prefix):
    for i, file_path in enumerate(os.listdir(source_dir), 1):
        _, ext = os.path.splitext(file_path)
        file_path = os.path.join(source_dir, file_path)

        ext = ext.lower()
        if ext in [".mov", ".mp4"]:
            audio = VideoFileClip(file_path).audio
        elif ext in [".m4a", ".ogg", ".mp3", ".wav"]:
            audio = AudioFileClip(file_path)
        else:
            continue

        audio_file_path = os.path.join(destination_dir, f"{filename_prefix}_{i}.wav")
        audio.write_audiofile(audio_file_path)


def resample(source_dir, destination_dir, filename_prefix):
    for i, audio_file in enumerate(glob.glob(os.path.join(source_dir, "*.wav"))):
        audio_segment: AudioSegment = AudioSegment.from_file(audio_file)

        if audio_segment.duration_seconds < 1:
            print(f"{audio_file} too short.")
            continue

        if audio_segment.channels != 1 or audio_segment.frame_rate != 16000:
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            print(f"{audio_file} fixed.")

        destination_file = os.path.join(destination_dir, f"{filename_prefix}_{i}.wav")
        audio_segment.export(destination_file, "wav")


def create_clips(source_dir, destination_dir, filename_prefix, clip_duration=1000):
    i = 1
    for audio_file in glob.glob(source_dir + "/*.wav"):
        audio_segment: AudioSegment = AudioSegment.from_file(audio_file)
        if len(audio_segment) < clip_duration:
            audio_segment += AudioSegment.silent(duration=clip_duration - len(audio_segment))

        clip_audio_file = os.path.join(destination_dir, filename_prefix + "_{i}.wav")
        for s in range(0, len(audio_segment) - clip_duration, clip_duration):
            audio_segment[s: s + clip_duration].export(clip_audio_file.format(i=i), format="wav")
            i += 1

        audio_segment[-clip_duration:].export(clip_audio_file.format(i=i), format="wav")
        i += 1
