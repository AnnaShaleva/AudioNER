from tensorflow.keras.models import model_from_json
from tensorflow import keras
import sys
import subprocess
import numpy as np
from tensorflow.keras.utils import to_categorical
import pandas as pd

import constants as const


def load_model(model_path, weights_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adam',
                  metrics=[keras.metrics.categorical_accuracy])
    print("Loaded model from disk")
    return  loaded_model

def load_dataset(dataset_name):
    print('Load data...')
    dataset_path = const.DATA_PATH + dataset_name + '/data_csv/'
    df = pd.read_csv(dataset_path + 'labels.csv', header=None)
    Y1 = df.values[:, 0]
    Y2 = df.values[:, 1]
    X = []
    for i in range(len(Y1)):
        df = pd.read_csv(dataset_path + str(i) + '.csv', header=None)
        X.append(df.values)
    print('%d samples were loaded', len(X))
    X = (np.array(X)).reshape(len(X), const.HEIGHT, const.LENGTH, 1)
    Y1 = to_categorical(Y1, const.N_CLASSES)
    Y2 = to_categorical(Y2, const.N_SUBCLASSES)
    return X, Y1, Y2

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    model_path = sys.argv[2]
    weights_path = sys.argv[3]
    model = load_model(model_path, weights_path)
    X, Y1, Y2 = load_dataset(dataset_name)

    _, _, _, cat_acc, subcat_acc = model.evaluate(X, [Y1, Y2], verbose=0)
    print('subcat_acc: %d', subcat_acc)
    print('cat_acc: %d', cat_acc)
