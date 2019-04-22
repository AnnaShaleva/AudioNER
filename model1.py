import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D
import sys

from load_data import get_test_and_train_data

print('Load data...')
X, Y1, Y2 = load_data('tiny_dataset')
print(len(X))
X = (np.array(X)).reshape(81, 20, 494, 1)

model = Sequential()
model.add(Conv2D(1, kernel_size = 256, padding = 'same', activation = 'relu', input_shape = (20, 494, 1)))
model.add(Conv2D(16, kernel_size = 128, padding = 'same', activation = 'relu'))
model.add(Conv2D(32, kernel_size = 64, padding = 'same', activation = 'relu'))
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