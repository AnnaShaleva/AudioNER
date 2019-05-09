import os
import sys
import constants as const
import scipy.io.wavfile
import numpy as np

def add_noise_to_audio(path, name):
    source_sample_path = path + name
    
    sr_value, x_value = scipy.io.wavfile.read(source_sample_path)
    target_snr = 100000
    sig_avg_x = np.mean(x_value)
    noise_avg_small = abs(sig_avg_x * target_snr)
    noise_avg_large = abs(sig_avg_x * target_snr)
    noise_small = np.random.normal(.0, np.sqrt(noise_avg_small), len(x_value)).round().astype(int)
    noise_large = np.random.normal(.0, np.sqrt(noise_avg_large), len(x_value)).round().astype(int)
    y_value = np.asarray(x_value + noise_small, dtype=np.int16)
    z_value = np.asarray(x_value + noise_large, dtype=np.int16)

    dest_sample_path = path + os.path.splitext(name)[0] + '_small_noised.wav'
    scipy.io.wavfile.write(dest_sample_path, sr_value, y_value)

    dest_sample_path = path + os.path.splitext(name)[0] + '_large_noised.wav'
    scipy.io.wavfile.write(dest_sample_path, sr_value, z_value)

if __name__=='__main__':
    dataset_name = sys.argv[1]
    dataset_path = os.path.join(const.DATA_PATH, dataset_name + '/', 'categories_samples/')
    for category_folder in os.listdir(dataset_path):
        category_source_path = dataset_path + category_folder + '/'
        for subcat_folder in os.listdir(category_source_path):
            subcat_source_path = category_source_path + subcat_folder + '/'
            for sample in os.listdir(subcat_source_path):
                add_noise_to_audio(subcat_source_path, sample)

