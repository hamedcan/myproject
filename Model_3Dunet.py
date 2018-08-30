from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras import losses
from keras import optimizers


def get_model(logger, log_disable, input_shape, pool_size=(2, 2), filter_size=(3, 3), n_labels=1):

    if log_disable == 0:
        logger.write('=============:model parameters:=============\n')
        logger.write('patch size: ' + str(input_shape) + '\n')
        logger.write('pool size: ' + str(pool_size) + '\n')
        logger.write('filter size: ' + str(filter_size) + '\n')

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1, activation='softmax'))

    model.compile(loss=losses.binary_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])
    return model
