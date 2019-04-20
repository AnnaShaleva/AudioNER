import os
import sys
import numpy as np

import constants as const
from normalize_audio import loud_norm, apply_bandpass_filter, correct_volume
from audio_file_to_input_vector import audiofile_to_input_vector

def load_data(dataset_name):
    dataset_path = os.path.join(const.DATA_PATH, dataset_name + '/', 'categories_samples/')

    if not os.path.isdir(dataset_path):
        raise Exception('Empty dataset folder')
    data = []
    category_label = 0
    for category_folder in os.listdir(dataset_path):
        subcategory_label = 0
        subcategory_path = os.path.join(dataset_path, category_folder)
        for subcategory_folder in os.listdir(subcategory_path):
            audio_path = os.path.join(subcategory_path, subcategory_folder)
            for audio_file in os.listdir(audio_path):
                audio = os.path.join(audio_path, audio_file)
                tmp_path1 = os.path.join(audio_path, 'tmp1.wav')
                tmp_path2 = os.path.join(audio_path, 'tmp2.wav')
                tmp_path3 = os.path.join(audio_path, 'tmp3.wav')
                loud_norm(audio, tmp_path1)
                apply_bandpass_filter(tmp_path1, tmp_path2)
                correct_volume(tmp_path2, tmp_path3)
                x = audiofile_to_input_vector(audio_filename=tmp_path3, numcep=const.N_INPUT, numcontext=const.N_CONTEXT)
                y1 = category_label
                y2 = subcategory_label
                data.append((x, y1, y2))
                os.remove(tmp_path1)
                os.remove(tmp_path2)
                os.remove(tmp_path3)
                print(x.shape)
            subcategory_label += 1
        category_label += 1
    np.random.shaffle(data)

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    load_data(dataset_name)
    #load_data('tiny_dataset')