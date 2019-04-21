import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D

from keras.layers import Dense, Flatten, Activation, Dropout
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
		self.conv1 = keras.layers.(Conv2D(1, 256, padding = 'same', activation = tf.nn.relu, input_shape = (20, 494, 1)))
		self.conv2 = keras.layers.(Conv2D(16, 128, padding = 'same', activation = tf.nn.relu))
		self.conv3 = keras.layers.(Conv2D(32, 64, padding = 'same', activation = tf.nn.relu))
		self.conv4 = keras.layers.(Conv2D(64, 32, padding = 'same', activation = tf.nn.relu))
		self.conv5 = keras.layers.(Conv2D(128, 16, padding = 'same', activation = tf.nn.relu))
		self.conv6 = keras.layers.(Conv2D(256, 8, activation = tf.nn.relu))
		self.flatten = keras.layers.Flatten()
		self.fc = keras.layaer.Dence(10, activation = tf.nn.relu, kernel_initializer = tf.initializer.variance_scaling, kernel_regulari)



print('Load data...')
X, Y1, Y2 = load_data('tiny_dataset')
print(len(X))
X = (np.array(X)).reshape(81, 20, 494, 1)
Y1 = to_categorical(Y1)
Y2 = to_categorical(Y2)
print('Building model')
model = Sequential()
model.add(Conv2D(1, kernel_size = 256, padding = 'same', activation = 'relu', input_shape = (20, 494, 1)))
model.add(Conv2D(16, kernel_size = 128, padding = 'same', activation = 'relu'))
model.add(Conv2D(32, kernel_size = 64, padding = 'same', activation = 'relu'))
model.add(Conv2D(64, kernel_size = 32, padding = 'same', activation = 'relu', ))
model.add(Conv2D(128, kernel_size = 16, padding = 'same', activation = 'relu'))
model.add(Conv2D(256, kernel_size = 8, activation = 'relu'))

model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

print(model.summary())

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print('Fitting model...')
results = model.fit(X, Y2, validation_split = 0.25, epochs = 10)
model.save_weights('model_weights.h5')
model.save('model.h5')
print(np.mean(results.history["val_acc"]))
