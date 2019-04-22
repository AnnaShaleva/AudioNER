import numpy as np
from keras.utils import to_categorical
#from keras.models import Sequential
#from keras.layers.convolutional import Conv2D

#from keras.layers import Dense, Flatten, Activation, Dropout
import tensorflow as tf
from tensorflow import keras

import sys

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import keras
import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 24} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

from load_data import load_data

class DigitsModel(keras.Model):
    def __init(self):
        super(DigitsModel, self).__init__(name='d_cnn')
        self.conv1 = keras.layers.Conv2D(1, 256, padding = 'same', activation = tf.nn.relu, input_shape = (20, 494, 1), kernel_initializer = tf.initializer.variance_scaling, kernel.regularizer=keras.regularizers.l2(l=0.001)))
        self.conv2 = keras.layers.Conv2D(16, 128, padding = 'same', activation = tf.nn.relu, kernel_initializer = tf.initializer.variance_scaling, kernel.regularizer=keras.regularizers.l2(l=0.001))
        self.conv3 = keras.layers.Conv2D(32, 64, padding = 'same', activation = tf.nn.relu, kernel_initializer = tf.initializer.variance_scaling, kernel.regularizer=keras.regularizers.l2(l=0.001))
        self.conv4 = keras.layers.Conv2D(64, 32, padding = 'same', activation = tf.nn.relu, kernel_initializer = tf.initializer.variance_scaling, kernel.regularizer=keras.regularizers.l2(l=0.001))
        self.conv5 = keras.layers.Conv2D(128, 16, padding = 'same', activation = tf.nn.relu, kernel_initializer = tf.initializer.variance_scaling, kernel.regularizer=keras.regularizers.l2(l=0.001))
        self.conv6 = keras.layers.Conv2D(256, 8, activation = tf.nn.relu, kernel_initializer = tf.initializer.variance_scaling, kernel.regularizer=keras.regularizers.l2(l=0.001))
        self.flatten = keras.layers.Flatten()
        self.fc = keras.layaer.Dence(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


print('Building model...')
model = DigitsModel()
model.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [
          # Write TensorBoard logs to `./logs` directory
            keras.callbacks.TensorBoard(log_dir='./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
            ]
print(model.summary())

print('Loading data...')
X, Y1, Y2 = load_data('tiny_dataset')
X = (np.array(X)).reshape(81, 20, 494, 1)
Y1 = to_categorical(Y1)
Y2 = to_categorical(Y2)

print('Fitting...')
model.fit(train_dataset, epochs=100, callbacks=callbacks)

#model = Sequential()
#model.add(Conv2D(1, kernel_size = 256, padding = 'same', activation = 'relu', input_shape = (20, 494, 1)))
#model.add(Conv2D(16, kernel_size = 128, padding = 'same', activation = 'relu'))
#model.add(Conv2D(32, kernel_size = 64, padding = 'same', activation = 'relu'))
#model.add(Conv2D(64, kernel_size = 32, padding = 'same', activation = 'relu', ))
#model.add(Conv2D(128, kernel_size = 16, padding = 'same', activation = 'relu'))
#model.add(Conv2D(256, kernel_size = 8, activation = 'relu'))

#model.add(Flatten())
#model.add(Dense(10, activation = 'softmax'))


#model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#print('Fitting model...')
#results = model.fit(X, Y2, validation_split = 0.25, epochs = 10)
model.save_weights('model_weights.h5')
model.save('model.h5')
#print(np.mean(results.history["val_acc"]))
