import os
import sys
import numpy as np
import csv
import pandas as pd

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
                if (x.shape[0] == 20) & (x.shape[1] == 494):
                    y1 = category_label
                    y2 = subcategory_label
                    data.append((x, y1, y2))
                else:
                    print(audio)
                    print(x.shape)
                os.remove(tmp_path1)
                os.remove(tmp_path2)
                os.remove(tmp_path3)
            subcategory_label += 1
        category_label += 1
    np.random.shuffle(data)
    reshaped_data = [list(a) for a in zip(*data)]
    X = reshaped_data[0]
    Y1 = reshaped_data[1]
    Y2 = reshaped_data[2]
    dataset_path = const.DATA_PATH + dataset_name + '/data_csv/'
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)
    df = pd.DataFrame({'Cat': Y1, 'Subcat': Y2})
    df.to_csv(dataset_path + 'labels.csv', header=None, index=None)
    i = 0
    for element in X:
        df = pd.DataFrame(element)
        df.to_csv(dataset_path + str(i) + '.csv', header=None, index=None)
        i += 1

    return X, Y1, Y2

def get_test_and_train_data(dataset_name, train_part):
    dataset_path = const.DATA_PATH + dataset_name + '/data_csv/'
    df = pd.read_csv(dataset_path + 'labels.csv', header=None)
    Y1 = df.values[:, 0]
    Y2 = df.values[:, 1]
    print(Y1)
    print(Y2)
    X = []
    for i in range(len(Y1)):
        df = pd.read_csv(dataset_path + str(i) + '.csv', header=None)
        X.append(df.values)
    #X, Y1, Y2 = load_data(dataset_name)
    train_num = int(len(X) * train_part)
    test_num = train_num - len(X)
    X_train = X[:train_num]
    X_test = X[test_num:]
    Y1_train = Y1[:train_num]
    Y1_test = Y1[test_num:]
    Y2_train = Y2[:train_num]
    Y2_test = Y2[test_num:]

    return X_train, Y1_train, Y2_train, X_test, Y1_test, Y2_test


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    load_data(dataset_name)
