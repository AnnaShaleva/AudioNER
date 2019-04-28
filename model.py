import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, UpSampling2D, MaxPooling2D
import sys

from load_data import load_data

from time import time
from tensorflow.python.keras.callbacks import TensorBoard

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#config = ConfigProto(device_count={'GPU':2, 'CPU':24})
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)
#keras.backend.set_session(session)
#session.as_default()

print('Building model...')
model = Sequential()
model.add(Conv2D(16, kernel_size = 3, activation = 'relu', padding='same', input_shape = (20, 494, 1)))
model.add(MaxPooling2D((2, 2),padding = 'same') )
model.add(Conv2D(32, kernel_size = 3, activation = 'relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64, kernel_size = 3, activation = 'relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(64, kernel_size = 3, activation = 'relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, kernel_size = 3, activation = 'relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, kernel_size = 3, activation = 'relu', padding='same'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(1, kernel_size = 3, activation = 'sigmoid', padding='same'))

#model.add(Conv2D(4, kernel_size = 3, activation = 'sigmoid'))
#model.add(Conv2D(1, kernel_size = 3, activation = 'sigmoid'))
#model.add(Conv2D(1, kernel_size = 3, activation = 'sigmoid'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128*20))
model.add(Dense(1280))
model.add(Dense(640))
model.add(Dense(8, activation = 'relu'))
print(model.summary())

tb = TensorBoard(log_dir='./log/{}'.format(time()))



model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


print('Load data...')
X, Y1, Y2 = load_data('test_dataset')
print(len(X))
X = (np.array(X)).reshape(len(X), 20, 494, 1)
Y1 = to_categorical(Y1)
Y2 = to_categorical(Y2)


print('Fitting model...')
results = model.fit(X, Y2, validation_split = 0.15, epochs = 100000, callbacks = [tb])
model.save_weights('model_weights.h5')
model.save('model.h5')

