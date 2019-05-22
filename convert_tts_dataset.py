import wave
import contextlib
import sys
import os
import subprocess
import constants as const


dataset_name = sys.argv[1]
dataset_path = os.path.join(const.DATA_PATH, dataset_name + '/', 'categories_samples/')

for category_folder in os.listdir(dataset_path):
        subcategory_path = os.path.join(dataset_path, category_folder)
        for subcategory_folder in os.listdir(subcategory_path):
            audio_path = os.path.join(subcategory_path, subcategory_folder)
            for audio_file in os.listdir(audio_path):
                audio = os.path.join(audio_path, audio_file)
                name = os.path.splitext(audio)[0] + '.wav'
                p = subprocess.Popen(["ffmpeg",
                    "-i", audio,
                    "-acodec", "pcm_s16le",
                    "-ac", "1",
                    "-ar", "16000",
                    "-filter:a", "atempo=1.25",
                    name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
                out, err = p.communicate()
                os.remove(audio)
                with contextlib.closing(wave.open(name, 'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    leng = frames/float(rate)
                    print(audio)
                    print(leng)

