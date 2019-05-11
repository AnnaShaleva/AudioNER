import os
import subprocess

for f in os.listdir('./categories_samples/'):
    audio_path = './categories_samples/' + f
    sample_path = os.path.splitext(f)[0] + '_.wav'
    p = subprocess.Popen(["ffmpeg",
                          "-i", audio_path,
                          "-acodec", "pcm_s16le",
                          "-ac", "1",
                          "-ar", "16000",
                          "-t", "0.8",
                          sample_path
                          ],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    out, err = p.communicate()

    if p.returncode != 0:
        raise Exception("Failed to cut audio at step 1: %s" % str(err))
