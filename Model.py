import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization
from keras import optimizers
from keras.layers.merge import concatenate
from keras_contrib.layers import Deconvolution3D

def get_model(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=True, downsize_filters_factor=4):
    """
    Builds the 3D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size).
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """

    print('=============:model parameters:=============')
    print('patch size: ', input_shape)
    print('downsize filters factor: ', downsize_filters_factor)
    print('learning rate: ', initial_learning_rate)
    print('pool size: ', pool_size)
    print('deconvolution: ', deconvolution)


    inputs = Input(input_shape)
    conv1 = Conv3D(int(32 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size, strides=2)(conv1)

    conv2 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size, strides=2)(conv2)

    conv3 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(int(256 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size, strides=2)(conv3)

    conv4 = Conv3D(int(256 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(int(512 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv4)

    up5 = UpSampling3D(size=pool_size)(conv4)
    up5 = Conv3D(int(256 / downsize_filters_factor), (2, 2, 2), padding='same')(up5)
    up5 = concatenate([up5, conv3], axis=4)
    conv5 = Conv3D(int(256 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(int(256 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = UpSampling3D(size=pool_size)(conv5)
    up6 = Conv3D(int(128 / downsize_filters_factor), (2, 2, 2), padding='same')(up6)
    up6 = concatenate([up6, conv2], axis=4)
    conv6 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = UpSampling3D(size=pool_size)(conv6)
    up7 = Conv3D(int(64 / downsize_filters_factor), (2, 2, 2), padding='same')(up7)
    up7 = concatenate([up7, conv1], axis=4)
    conv7 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv7)

    conv8 = Conv3D(n_labels, (1, 1, 1))(conv7)
    act = Activation('sigmoid')(conv8)
    model = Model(inputs=inputs, outputs=act)

    adam = optimizers.Adam(lr=initial_learning_rate)

    model.compile(optimizer=adam, loss=dice_coef_loss , metrics=[dice_coef, 'acc'])
    return model


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def compute_level_output_shape(filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node
    """
    if depth != 0:
        output_image_shape = np.divide(image_shape, np.multiply(pool_size, depth)).tolist()
    else:
        output_image_shape = image_shape
    return tuple([None, filters] + [int(x) for x in output_image_shape])


def get_upconv(depth, nb_filters, pool_size, image_shape, kernel_size=(2, 2, 2), strides=(2, 2, 2),
               deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=nb_filters, kernel_size=kernel_size,
                               output_shape=compute_level_output_shape(filters=nb_filters, depth=depth,
                                                                       pool_size=pool_size,
                                                                       image_shape=image_shape),
                               strides=strides, input_shape=compute_level_output_shape(filters=nb_filters,
                                                                                       depth=depth + 1,
                                                                                       pool_size=pool_size,
                                                                                       image_shape=image_shape))
    else:
        return UpSampling3D(size=pool_size)
