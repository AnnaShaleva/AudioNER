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
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D, UpSampling2D
import nni
sys.path.insert(0, '/headless/shared/AudioNER/')

import constants as const
from load_data import get_test_and_train_data


LOG = logging.getLogger('deep_conv__for_noise2x_data_model_keras')
TENSORBOARD_DIR = os.environ['NNI_OUTPUT_DIR']


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


def build_model(hyper_params, input_shape=(const.HEIGHT, const.LENGTH, 1), num_classes=const.N_CLASSES, num_subclasses=const.N_SUBCLASSES):
    print('Building model...')
    #input layer
    visible = Input(shape=input_shape)
    conv0 = Conv2D(hyper_params['conv0_size'], kernel_size=(8, 10), use_bias=True, padding='same', activation='relu')(visible)

    #classification layers
    pool1 = MaxPooling2D((2, 2), padding='same')(conv0)
    conv1 = Conv2D(hyper_params['conv1_size'], kernel_size=(4, 10), use_bias=True, padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D((2, 2), padding='same')(conv1)
    conv2 = Conv2D(hyper_params['conv2_size'], kernel_size=(2, 5), use_bias=True, padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D((2, 2), padding='same')(conv2)
    conv3 = Conv2D(hyper_params['conv3_size'], kernel_size=(1, 1), use_bias=True, padding='same', activation='relu')(pool3)
    pool4 = MaxPooling2D((2, 2), padding='same')(conv3)

    flatten = Flatten()(pool4)

    classes_output = Dense(num_classes, activation='softmax')(flatten)

    # subclasses_output
    dense1 = Dense(hyper_params['dense_size'], activation='relu')(flatten)
    subclasses_output = Dense(num_subclasses, activation='softmax')(dense1)

    model = Model(inputs=visible, outputs=[classes_output, subclasses_output])

    # summarize layers
    print(model.summary())

    # plot graph
    #plot_model(model, to_file='multiple_outputs.png')

    if hyper_params['optimizer'] == 'Adam':
        optimizer = keras.optimizers.Adam(lr=hyper_params['learning_rate'])
    else:
        optimizer = keras.optimizers.SGD(lr=hyper_params['learning_rate'], momentum=0.9)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=[keras.metrics.categorical_accuracy])
    print(model.metrics_names)
    return model


class SendMetrics(keras.callbacks.Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''
    def on_epoch_end(self, epoch, logs={}):
        '''
        Run on end of each epoch
        '''
        LOG.debug(logs)        
        nni.report_intermediate_result(logs["dense_2_categorical_accuracy"])


def train(args, params):
    '''
    Train model
    '''
    model = build_model(params)
    X_train, Y1_train, Y2_train, X_test, Y1_test, Y2_test = load_dataset(args.dataset_name, args.num_train)

    print('Fitting model...')
    results = model.fit(X_train, [Y1_train, Y2_train], epochs=args.epochs, verbose=1,
                        validation_data=(X_test, [Y1_test, Y2_test]), callbacks=[SendMetrics(), TensorBoard(log_dir=TENSORBOARD_DIR)])

    _, _, _, cat_acc, subcat_acc = model.evaluate(X_test, [Y1_test, Y2_test], verbose=0)
    LOG.debug('Final result is: %d', subcat_acc)
    nni.report_final_result(subcat_acc)
    print('Final result is: %d', subcat_acc)

    model_id = nni.get_sequence_id()
    # serialize model to JSON
    model_json = model.to_json()
    with open("model-{}.json".format(model_id), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model-{}.h5".format(model_id))
    print("Saved model to disk")


def generate_default_params():
    '''
    Generate default hyper parameters
    '''
    return {
        'optimizer': 'Adam',
        'learning_rate': 0.001
    }

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--dataset_name", type=str,default='test_noised2x_data', help='Dataset name', required=False)
    PARSER.add_argument("--epochs", type=int, default=400, help="Train epochs", required=False)
    PARSER.add_argument("--num_train", type=float, default=0.8,
                        help="Part of train samples to be used, maximum 1", required=False)

    ARGS, UNKNOWN = PARSER.parse_known_args()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = generate_default_params()
        PARAMS.update(RECEIVED_PARAMS)
        # train
        train(ARGS, PARAMS)
    except Exception as e:
        LOG.exception(e)
        raise
