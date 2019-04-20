import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras import layers
import sys

from load_data import get_test_and_train_data

X_train, Y1_train, Y2_train, X_test, Y1_test, Y2_test = get_test_and_train_data('tiny_dataset', 0.75)
Y1_train = to_categorical(Y1_train)
Y2_train = to_categorical(Y2_train)
Y1_test = to_categorical(Y1_test)
Y2_test = to_categorical(Y2_test)

model = Sequential()
model.add(Conv2D(64, kernel_size = 3, activation = 'relu', input_shape = (40, 494)))
model.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(4, activation = 'softmax'))

model.summary()

model,compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

results = model.fit(X_train, Y2_train, validation_data = (X_test, Y2_test), epochs = 3)

print(np.mean(results.history["val_acc"]))