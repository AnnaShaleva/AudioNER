import os
import sys
import constants as const
import scipy.io.wavfile
import numpy as np
import subprocess


def speed_up_audio(path, name):
    source_sample_path = path + name
    dest_path = path + "faster.wav"
    p = subprocess.Popen(["ffmpeg",
                          "-y",
                          "-i", source_sample_path,
                          "-acodec", "pcm_s16le",
                          "-ac", "1",
                          "-ar", "16000",
                          "-filter:a", "atempo=1.25",
                          dest_path
                          ],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    out, err = p.communicate()

    if p.returncode != 0:
        raise Exception("Failed to speed up audio: %s" % str(err))

    sr_value, x_value = scipy.io.wavfile.read(dest_path)
    os.remove(dest_path)

    target_snr = 100
    sig_avg_x = np.mean(x_value)
    noise_avg = abs(sig_avg_x * target_snr)
    dim_start = int((12800 - len(x_value)) / 2)
    dim_end = 12800 - dim_start - len(x_value)
    noise_start = np.random.normal(.0, np.sqrt(noise_avg), dim_start).round().astype(int)
    noise_end = np.random.normal(.0, np.sqrt(noise_avg), dim_end).round().astype(int)
    result = np.concatenate((noise_start, x_value, noise_end))
    y_value = np.asarray(result, dtype=np.int16)
    dest_sample_path = path + os.path.splitext(name)[0] + '_faster.wav'
    scipy.io.wavfile.write(dest_sample_path, sr_value, y_value)


def add_noise_to_audio(path, name):
    source_sample_path = path + name
    
    sr_value, x_value = scipy.io.wavfile.read(source_sample_path)
    #target_snr_low = 2
    target_snr_large = 2
    sig_avg_x = np.mean(abs(x_value))
    #noise_avg_small = sig_avg_x * target_snr_low
    noise_avg_large = sig_avg_x * target_snr_large
    #noise_small = np.random.normal(.0, np.sqrt(noise_avg_small), len(x_value)).round().astype(int)
    noise_large = np.random.normal(.0, np.sqrt(noise_avg_large), len(x_value)).round().astype(int)
    #y_value = np.asarray(x_value + noise_small, dtype=np.int16)
    z_value = np.asarray(x_value + noise_large, dtype=np.int16)

    #dest_sample_path = path + os.path.splitext(name)[0] + '_small_noised.wav'
    #scipy.io.wavfile.write(dest_sample_path, sr_value, y_value)

    dest_sample_path = path + os.path.splitext(name)[0] + '_large_noised.wav'
    scipy.io.wavfile.write(dest_sample_path, sr_value, z_value)
    os.remove(source_sample_path)



if __name__=='__main__':
    dataset_name = sys.argv[1]
    dataset_path = os.path.join(const.DATA_PATH, dataset_name + '/', 'categories_samples/')
    for category_folder in os.listdir(dataset_path):
        category_source_path = dataset_path + category_folder + '/'
        for subcat_folder in os.listdir(category_source_path):
            subcat_source_path = category_source_path + subcat_folder + '/'
            #for sample in os.listdir(subcat_source_path):
            #    speed_up_audio(subcat_source_path, sample)
            for sample in os.listdir(subcat_source_path):
                add_noise_to_audio(subcat_source_path, sample)

