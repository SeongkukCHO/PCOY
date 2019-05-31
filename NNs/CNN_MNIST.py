import sys
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
np.random.seed(2019)
(tr_X, tr_Y), (te_X, te_Y) = keras.datasets.mnist.load_data()

input_shape = (28, 28, 1)
tr_X = tr_X.reshape(tr_X.shape[0], 28, 28, 1)
te_X = te_X.reshape(te_X.shape[0], 28, 28, 1)

tr_X = tr_X.astype('float32') / 255.0
te_X = te_X.astype('float32') / 255.0

batch_size = 100
num_classes = 10
epochs = 12

tr_Y = keras.utils.to_categorical(tr_Y, num_classes)
te_Y = keras.utils.to_categorical(te_Y, num_classes)
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(tr_X, tr_Y,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1, 
                 validation_data=(te_X, te_Y))
                 score = model.evaluate(te_X, te_Y, verbose=0)
print('loss:', score[0])
print('accuracy:', score[1])
