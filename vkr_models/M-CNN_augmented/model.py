import sys
import os
import argparse
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, SeparableConv2D, Activation, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
sys.path.insert(0, '/headless/shared/AudioNER/')

import constants as const
from load_data import get_test_and_train_data



TENSORBOARD_DIR = '/headless/shared/AudioNER/vkr_models/M-CNN_augmented/log/'

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
    visible = Input(shape=input_shape)
    
    #classification layers
    conv0 = Conv2D(64, kernel_size=(8, 10), use_bias=True, padding='same', name='Conv_0')(visible)
    batch_normalization0 = BatchNormalization(name='Batch_normalization_0')(conv0)
    activation0 = Activation('relu', name='ReLU_0')(batch_normalization0)
    pool0 = MaxPooling2D((2, 2), padding='same', name='MaxPooling_0')(activation0)
    dropout0 = Dropout(0.2, name='Dropout_0')(pool0)

    conv1 = Conv2D(16, kernel_size=(4, 10), use_bias=True, padding='same', name='Conv_1')(dropout0)
    batch_normalization1 = BatchNormalization(name='Batch_normalization_1')(conv1)
    activation1 = Activation('relu', name='ReLU_1')(batch_normalization1)
    pool1 = MaxPooling2D((2, 2), padding='same', name='MaxPooling_1')(activation1)
    dropout1 = Dropout(0.2, name='Dropout_1')(pool1)

    conv2 = Conv2D(128, kernel_size=(2, 5), use_bias=True, padding='same', name='Conv_2')(dropout1)
    batch_normalization2 = BatchNormalization(name='Batch_normalization_2')(conv2)
    activation2 = Activation('relu', name='ReLU_2')(batch_normalization2)
    dropout2 = Dropout(0.2, name='Dropout_2')(activation2)

    conv3 = Conv2D(16, kernel_size=(1, 1), use_bias=True, padding='same', name='Conv_3')(dropout2)
    batch_normalization3 = BatchNormalization(name='Batch_normalization_3')(conv3)
    activation3 = Activation('relu', name='ReLU_3')(batch_normalization3)
    dropout3 = Dropout(0.2, name='Dropout_3')(activation3)

    concat = concatenate([dropout3, dropout1])

    flatten = Flatten(name='Flatten')(concat)

    classes_output = Dense(num_classes, activation='softmax', name='Classes_output')(flatten)

    # subclasses_output
    dense1 = Dense(256, activation='relu', name='Dense')(flatten)
    subclasses_output = Dense(num_subclasses, activation='softmax', name='Subclasses_output')(dense1)


    model = Model(inputs=visible, outputs=[classes_output, subclasses_output])

    # summarize layers
    print(model.summary())

    optimizer = keras.optimizers.SGD(lr=0.0005, momentum=0.9)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=[keras.metrics.categorical_accuracy])
    print(model.metrics_names)
    return model

def train(args):
    '''
    Train model
    '''
    model = build_model()
    X_train, Y1_train, Y2_train, X_test, Y1_test, Y2_test = load_dataset(args.dataset_name, args.num_train)

    mc = ModelCheckpoint('best_model.h5', monitor='dense_2_categorical_accuracy', mode='max', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

    print('Fitting model...')
    results = model.fit(X_train, [Y1_train, Y2_train], epochs=args.epochs, verbose=1,
                        validation_data=(X_test, [Y1_test, Y2_test]), callbacks=[TensorBoard(log_dir=TENSORBOARD_DIR), es, mc])

    _, _, _, cat_acc, subcat_acc = model.evaluate(X_test, [Y1_test, Y2_test], verbose=0)
    print('Final result is: %d', subcat_acc)

    model.save("model.h5")
    print("Saved model to disk")


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
