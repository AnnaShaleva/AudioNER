import sys
import os
import argparse
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
from keras.utils import to_categorical, plot_model
from keras.models import Model
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D, UpSampling2D
import nni

import constants as const
from load_data import get_test_and_train_data



TENSORBOARD_DIR = './log/'


def load_dataset(dataset_name, part_train):
    print('Load data...')
    X_train, Y1_train, Y2_train, X_test, Y1_test, Y2_test = get_test_and_train_data(dataset_name, part_train)
    print('%d train samples were loaded', len(X_train))
    print('%d test samples were loaded', len(X_test))
    X_train = (np.array(X_train)).reshape(len(X_train), const.HEIGHT, const.LENGTH, 1)
    X_test = (np.array(X_test)).reshape(len(X_test), const.HEIGHT, const.LENGTH, 1)
    Y1_train = to_categorical(Y1_train, const.N_CLASSES)
    Y1_test = to_categorical(Y1_test, const.N_CLASSES)
    Y2_train = to_categorical(Y2_train, const.N_SUBCLASSES)
    Y2_test = to_categorical(Y2_test, const.N_SUBCLASSES)
    return X_train, Y1_train, Y2_train, X_test, Y1_test, Y2_test


def build_model(input_shape=(const.HEIGHT, const.LENGTH, 1), num_classes=const.N_CLASSES, num_subclasses=const.N_SUBCLASSES):
    print('Building model...')
    #input layer
    visible = Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=input_shape)

    #classification layers
    #encoder
    pool1 = MaxPooling2D((2, 2), padding='same')(visible)
    conv1 = Conv2D(32, kernel_size=3, padding = 'same', activation = 'relu')(pool1)
    pool2 = MaxPooling2D((2, 2), padding='same')(conv1)
    conv2 = Conv2D(64, kernel_size=3, padding = 'same', activation = 'relu')(pool2)
    pool3 = MaxPooling2D((2, 2), padding='same')(conv2)

    #decoder
    conv3 = Conv2D(64, kernel_size=3, padding = 'same', activation = 'relu')(pool3)
    unpool1 = UpSampling2D((2,2))(conv3)
    conv4 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(unpool1)
    unpool2 = UpSampling2D((2, 2))(conv4)
    conv5 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(unpool2)
    unpool2 = UpSampling2D((2, 2))(conv5)
    conv6 = Conv2D(1, kernel_size=3, padding='same', activation = 'sigmoid')(unpool2)
    dropout = Dropout(0.1)(conv6)
    flatten = Flatten()(dropout)

    #classes output
    classes_output = Dense(num_classes, activation = 'softmax')(flatten)

    #subclasses_output
    dense1 = Dense(256)(flatten)
    dense2 = Dense(128)(dense1)
    dense3 = Dense(64)(dense2)
    subclasses_output = Dense(num_subclasses, activation = 'softmax')(dense3)

    model = Model(input=visible, outputs=[classes_output, subclasses_output])

    # summarize layers
    print(model.summary())

    # plot graph
    plot_model(model, to_file='multiple_outputs.png')

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
    return model


def train(args):
    '''
    Train model
    '''
    model = build_model()
    X_train, Y1_train, Y2_train, X_test, Y1_test, Y2_test = load_dataset(args.dataset_name, args.num_train)

    print('Fitting model...')
    results = model.fit(X_train, [Y1_train, Y2_train], batch_size=args.batch_size, epochs=args.epochs, verbose=1,
                        validation_data=(X_test, [Y1_test, Y2_test]), callbacks=[TensorBoard(log_dir=TENSORBOARD_DIR)])

    _, acc = model.evaluate(X_test, [Y1_test, Y2_test], verbose=0)
    print('Final result is: %d', acc)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--dataset_name", type=str, help='Dataset name', required=True)
    PARSER.add_argument("--epochs", type=int, default=1000, help="Train epochs", required=False)
    PARSER.add_argument("--num_train", type=float, default=0.8,
                        help="Part of train samples to be used, maximum 1", required=False)

    ARGS, UNKNOWN = PARSER.parse_known_args()

    train(ARGS)
