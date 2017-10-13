from __future__ import print_function

import keras
import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten, Conv3D, MaxPooling3D
from keras.models import Sequential

from DS import DS

# initialization and prepare data set#########################################################
patch_size = 5
radius = 20
batch_size = 32
epochs = 2
max_gray_level = 4096
testrate = 0.2

ds = DS('.\data\\', testrate, max_gray_level, patch_size, radius)
x_train, y_train, x_test, y_test = ds.get_data()
# define mode#################################################################################
model = Sequential()
model.add(Conv3D(4, (3, 3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv3D(4, (3, 3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))

model.add(Conv3D(8, (3, 3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv3D(8, (3, 3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))
# train model############################################################################################
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)
# get accuracy on test data############################################################################
classes = model.predict(x_test)

y_true = np.zeros(y_test.shape[0])
for i in range(len(y_true)):
    if classes[i, 0] <= classes[i, 1]:
        y_true[i] = 1

y_predicted = np.zeros_like(y_true)
for i in range(y_test.shape[0]):
    if y_test[i, 0] <= y_test[i, 1]:
        y_predicted[i] = 1


y_true = np.zeros(y_test.shape[0])
for i in range(len(y_true)):
    if classes[i, 0] <= classes[i, 1]:
        y_true[i] = 1