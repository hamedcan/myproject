import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, Deconv3D, BatchNormalization
from keras import optimizers
from keras.layers.merge import concatenate
import tensorflow as tf
from Init_data import InitData


def get_model(logger, log_disable, input_shape, pool_size=(2, 2, 2), filter_size=(3, 3, 3), n_labels=1):
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
    if log_disable == 0:
        logger.write('=============:model parameters:=============\n')
        logger.write('patch size: ' + str(input_shape) + '\n')
        logger.write('pool size: ' + str(pool_size) + '\n')
        logger.write('filter size: ' + str(filter_size) + '\n')

    inputs = Input(input_shape)
    conv1 = Conv3D(16, filter_size, padding='same', activation='relu')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(32, filter_size, padding='same', activation='relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size, strides=2)(conv1)

    conv2 = Conv3D(32, filter_size, padding='same', activation='relu')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(64, filter_size, padding='same', activation='relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size, strides=2)(conv2)

    conv3 = Conv3D(64, filter_size, padding='same', activation='relu')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(128, filter_size, padding='same', activation='relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size, strides=2)(conv3)

    conv4 = Conv3D(128, filter_size, padding='same', activation='relu')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv3D(256, filter_size, padding='same', activation='relu')(conv4)
    conv4 = BatchNormalization()(conv4)

    # up5 = Deconv3D(128, filter_size, strides=(2, 2, 2), padding='same', activation='relu')(conv4)
    up5 = UpSampling3D(size=pool_size)(conv4)
    up5 = Conv3D(128, filter_size, padding='same', activation='relu')(up5)
    up5 = BatchNormalization()(up5)
    up5 = concatenate([up5, conv3], axis=4)
    conv5 = Conv3D(128, filter_size, padding='same', activation='relu')(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3D(128, filter_size, padding='same', activation='relu')(conv5)
    conv5 = BatchNormalization()(conv5)

    # up6 = Deconv3D(64, filter_size, strides=(2,2,2), padding='same', activation='relu')(conv5)
    up6 = UpSampling3D(size=pool_size)(conv5)
    up6 = Conv3D(64, filter_size, padding='same', activation='relu')(up6)
    up6 = BatchNormalization()(up6)
    up6 = concatenate([up6, conv2], axis=4)
    conv6 = Conv3D(64, filter_size, padding='same', activation='relu')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv3D(64, filter_size, padding='same', activation='relu')(conv6)
    conv6 = BatchNormalization()(conv6)

    # up7 = Deconv3D(32, filter_size, strides=(2, 2, 2), padding='same', activation='relu')(conv6)
    up7 = UpSampling3D(size=pool_size)(conv6)
    up7 = Conv3D(32, filter_size, padding='same', activation='relu')(up7)
    up7 = BatchNormalization()(up7)
    up7 = concatenate([up7, conv1], axis=4)
    conv7 = Conv3D(32, filter_size, padding='same', activation='relu')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv3D(32, filter_size, padding='same', activation='relu')(conv7)
    conv7 = BatchNormalization()(conv7)

    conv8 = Conv3D(n_labels, (1, 1, 1))(conv7)
    act = Activation('sigmoid')(conv8)
    model = Model(inputs=inputs, outputs=act)

    adam = optimizers.Adam(lr=0.001)

    model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef, 'acc'])
    return model


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def tao_method(shape, dtype='float32'):
    data = InitData.__call__()
    print("initializing with size: " + str(shape))
    return tf.convert_to_tensor(np.array(data.get(shape[3], shape[4])), dtype=dtype)
