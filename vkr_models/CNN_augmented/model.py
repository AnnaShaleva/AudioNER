import sys
import os
import argparse
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
sys.path.insert(0, '/headless/shared/AudioNER/')

import constants as const
from load_data import get_train_dev_test_data


TENSORBOARD_DIR = '/headless/shared/AudioNER/vkr_models/CNN_augmented_3/log/'


def load_dataset(dataset_name):
    print('Load data...')
    X_train, Y1_train, Y2_train, X_dev, Y1_dev, Y2_dev, X_test, Y1_test, Y2_test = get_train_dev_test_data(dataset_name)
    print('%d train samples were loaded', len(X_train))
    print('%d test samples were loaded', len(X_test))
    X_train = (np.array(X_train)).reshape(len(X_train), const.HEIGHT, const.LENGTH, 1)
    X_dev = (np.array(X_dev)).reshape(len(X_dev), const.HEIGHT, const.LENGTH, 1)
    X_test = (np.array(X_test)).reshape(len(X_test), const.HEIGHT, const.LENGTH, 1)
    Y1_train = to_categorical(Y1_train, const.N_CLASSES)
    Y1_dev = to_categorical(Y1_dev, const.N_CLASSES)
    Y1_test = to_categorical(Y1_test, const.N_CLASSES)
    Y2_train = to_categorical(Y2_train, const.N_SUBCLASSES)
    Y2_dev = to_categorical(Y2_dev, const.N_SUBCLASSES)
    Y2_test = to_categorical(Y2_test, const.N_SUBCLASSES)
    return X_train, Y1_train, Y2_train, X_dev, Y1_dev, Y2_dev, X_test, Y1_test, Y2_test


def build_model(input_shape=(const.HEIGHT, const.LENGTH, 1), num_classes=const.N_CLASSES, num_subclasses=const.N_SUBCLASSES):
    print('Building model...')
    #input layer
    visible = Input(shape=input_shape)
    
    #classification layers
    conv0 = Conv2D(256, kernel_size=(8, 10), use_bias=True, padding='same')(visible)
    batch_normalization0 = BatchNormalization()(conv0)
    activation0 = Activation('relu')(batch_normalization0)
    pool0 = MaxPooling2D((2, 2), padding='same')(activation0)
    dropout0 = Dropout(0.2)(pool0)

    conv1 = Conv2D(256, kernel_size=(4, 10), use_bias=True, padding='same')(dropout0)
    batch_normalization1 = BatchNormalization()(conv1)
    activation1 = Activation('relu')(batch_normalization1)
    pool1 = MaxPooling2D((2, 2), padding='same')(activation1)
    dropout1 = Dropout(0.2)(pool1)

    conv2 = Conv2D(64, kernel_size=(2, 5), use_bias=True, padding='same')(dropout1)
    batch_normalization2 = BatchNormalization()(conv2)
    activation2 = Activation('relu')(batch_normalization2)
    pool2 = MaxPooling2D((2, 2), padding='same')(activation2)
    dropout2 = Dropout(0.2)(pool2)

    conv3 = Conv2D(16, kernel_size=(1, 1), use_bias=True, padding='same')(dropout2)
    batch_normalization3 = BatchNormalization()(conv3)
    activation3 = Activation('relu')(batch_normalization3)
    pool3 = MaxPooling2D((2, 2), padding='same')(activation3)
    dropout3 = Dropout(0.2)(pool3)

    flatten = Flatten()(dropout3)

    classes_output = Dense(num_classes, activation='softmax')(flatten)

    # subclasses_output
    dense1 = Dense(128, activation='relu')(flatten)
    subclasses_output = Dense(num_subclasses, activation='softmax')(dense1)

    model = Model(inputs=visible, outputs=[classes_output, subclasses_output])

    # summarize layers
    print(model.summary())
    optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=[keras.metrics.categorical_accuracy])
    print(model.metrics_names)
    return model

def train(args):
    '''
    Train model
    '''
    model = build_model()
    X_train, Y1_train, Y2_train, X_dev, Y1_dev, Y2_dev, X_test, Y1_test, Y2_test = load_dataset(args.dataset_name)
    
    mc = ModelCheckpoint('best_model.h5', monitor='dense_2_categorical_accuracy', mode='max', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

    print('Fitting model...')
    results = model.fit(X_train, [Y1_train, Y2_train], epochs=args.epochs, verbose=1,
                        validation_data=(X_dev, [Y1_dev, Y2_dev]), callbacks=[TensorBoard(log_dir=TENSORBOARD_DIR), es, mc])

    _, _, _, cat_acc, subcat_acc = model.evaluate(X_test, [Y1_test, Y2_test], verbose=0)
    print('last model:')
    print(cat_acc)
    print(subcat_acc)
    model.save("model.h5")
    print("Saved model to disk")

    model1 = load_model('best_model.h5')
    _, _, _, cat_acc, subcat_acc = model1.evaluate(X_test, [Y1_test, Y2_test], verbose=0)
    print('best model:')
    print(cat_acc)
    print(subcat_acc)



if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--dataset_name", type=str,default='vkr_dataset_augmented', help='Dataset name', required=False)
    PARSER.add_argument("--epochs", type=int, default=300, help="Train epochs", required=False)
    PARSER.add_argument("--num_train", type=float, default=0.8,
                        help="Part of train samples to be used, maximum 1", required=False)

    ARGS, UNKNOWN = PARSER.parse_known_args()

    try:
        # train
        train(ARGS)
    except Exception as e:
        raise
