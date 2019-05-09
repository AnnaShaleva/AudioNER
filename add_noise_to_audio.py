import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import subprocess

sr_value, x_value = scipy.io.wavfile.read("1214.wav")
print(sr_value)
print(x_value.shape)

p = subprocess.Popen(["ffmpeg",
                      "-y",
            "-i", "1214.wav",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            "-filter:a", "atempo=1.25",
            "faster.wav"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

out, err = p.communicate()


if p.returncode != 0:
   raise Exception("Failed to cut audio: %s" % str(err))

sr_value, x_value = scipy.io.wavfile.read("faster.wav")
target_snr = 10
sig_avg_x = abs(np.mean(x_value))
noise_avg_x = sig_avg_x * target_snr
dim_start = int((12800 - len(x_value)) / 2)
print(dim_start)
dim_end = 12800 - dim_start - len(x_value)
noise_start = np.random.normal(.0, np.sqrt(noise_avg_x), dim_start ).round().astype(int)
noise_end = np.random.normal(.0, np.sqrt(noise_avg_x), dim_end ).round().astype(int)
result = np.concatenate((noise_start, x_value, noise_end))
y_value = np.asarray(result, dtype=np.int16)
scipy.io.wavfile.write("faster_with_noise.wav", sr_value, y_value)
#
# sr_value, x_value = scipy.io.wavfile.read("cut.wav")
# print(sr_value)
# print(x_value.shape)
# target_snr = 1000000
# sig_avg_x = np.mean(x_value)
# print(sig_avg_x)
# noise_avg_x = sig_avg_x * target_snr
# print(noise_avg_x)
# noise = np.random.normal(.0, np.sqrt(noise_avg_x), len(x_value)).round().astype(int)
# y_value = np.asarray(x_value + noise, dtype=np.int16)
# scipy.io.wavfile.write("cut_with_noise.wav", sr_value, y_value)

