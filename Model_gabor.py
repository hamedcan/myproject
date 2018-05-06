import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, Deconv3D, BatchNormalization, Lambda
from keras import optimizers
from keras.layers.merge import concatenate
import tensorflow as tf
from Init_data_gabor import InitData


def get_model(logger, log_disable, input_shape, pool_size=(2, 2, 2), filter_size=(3, 3, 3), n_labels=1):
    init_data = InitData.__call__()
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

    conv1_1 = Conv3D(16, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(crop(0)(inputs))
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = Conv3D(32, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(conv1_1)
    conv1_1 = BatchNormalization()(conv1_1)
    pool1_1 = MaxPooling3D(pool_size=pool_size, strides=2)(conv1_1)

    conv2_1 = Conv3D(32, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(pool1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = Conv3D(64, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(conv2_1)
    conv2_1 = BatchNormalization()(conv2_1)
    pool2_1 = MaxPooling3D(pool_size=pool_size, strides=2)(conv2_1)

    conv3_1 = Conv3D(64, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(pool2_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = Conv3D(128, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(conv3_1)
    conv3_1 = BatchNormalization()(conv3_1)
    pool3_1 = MaxPooling3D(pool_size=pool_size, strides=2)(conv3_1)

    conv4_1 = Conv3D(128, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(pool3_1)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_1 = Conv3D(256, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(conv4_1)
    conv4_1 = BatchNormalization()(conv4_1)

    up5_1 = UpSampling3D(size=pool_size)(conv4_1)
    up5_1 = Conv3D(256, (2, 2, 2), padding='same', activation='relu', kernel_initializer=tao_method)(up5_1)
    up5_1 = BatchNormalization()(up5_1)
    up5_1 = concatenate([up5_1, conv3_1], axis=4)
    conv5_1 = Conv3D(128, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(up5_1)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_1 = Conv3D(128, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(conv5_1)
    conv5_1 = BatchNormalization()(conv5_1)

    up6_1 = UpSampling3D(size=pool_size)(conv5_1)
    up6_1 = Conv3D(128, (2, 2, 2), padding='same', activation='relu', kernel_initializer=tao_method)(up6_1)
    up6_1 = BatchNormalization()(up6_1)
    up6_1 = concatenate([up6_1, conv2_1], axis=4)
    conv6_1 = Conv3D(64, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(up6_1)
    conv6_1 = BatchNormalization()(conv6_1)
    conv6_1 = Conv3D(64, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(conv6_1)
    conv6_1 = BatchNormalization()(conv6_1)

    up7_1 = UpSampling3D(size=pool_size)(conv6_1)
    up7_1 = Conv3D(64, (2, 2, 2), padding='same', activation='relu', kernel_initializer=tao_method)(up7_1)
    up7_1 = BatchNormalization()(up7_1)
    up7_1 = concatenate([up7_1, conv1_1], axis=4)
    conv7_1 = Conv3D(32, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(up7_1)
    conv7_1 = BatchNormalization()(conv7_1)
    conv7_1 = Conv3D(32, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(conv7_1)
    conv7_1 = BatchNormalization()(conv7_1)

    conv1_2 = Conv3D(16, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(crop(1)(inputs))
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = Conv3D(32, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(conv1_2)
    conv1_2 = BatchNormalization()(conv1_2)
    pool1_2 = MaxPooling3D(pool_size=pool_size, strides=2)(conv1_2)

    conv2_2 = Conv3D(32, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(pool1_2)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = Conv3D(64, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(conv2_2)
    conv2_2 = BatchNormalization()(conv2_2)
    pool2_2 = MaxPooling3D(pool_size=pool_size, strides=2)(conv2_2)

    conv3_2 = Conv3D(64, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(pool2_2)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_2 = Conv3D(128, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(conv3_2)
    conv3_2 = BatchNormalization()(conv3_2)
    pool3_2 = MaxPooling3D(pool_size=pool_size, strides=2)(conv3_2)

    conv4_2 = Conv3D(128, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(pool3_2)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_2 = Conv3D(256, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(conv4_2)
    conv4_2 = BatchNormalization()(conv4_2)

    up5_2 = UpSampling3D(size=pool_size)(conv4_2)
    up5_2 = Conv3D(256, (2, 2, 2), padding='same', activation='relu', kernel_initializer=tao_method)(up5_2)
    up5_2 = BatchNormalization()(up5_2)
    up5_2 = concatenate([up5_2, conv3_2], axis=4)
    conv5_2 = Conv3D(128, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(up5_2)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_2 = Conv3D(128, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(conv5_2)
    conv5_2 = BatchNormalization()(conv5_2)

    up6_2 = UpSampling3D(size=pool_size)(conv5_2)
    up6_2 = Conv3D(128, (2, 2, 2), padding='same', activation='relu', kernel_initializer=tao_method)(up6_2)
    up6_2 = BatchNormalization()(up6_2)
    up6_2 = concatenate([up6_2, conv2_2], axis=4)
    conv6_2 = Conv3D(64, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(up6_2)
    conv6_2 = BatchNormalization()(conv6_2)
    conv6_2 = Conv3D(64, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(conv6_2)
    conv6_2 = BatchNormalization()(conv6_2)

    up7_2 = UpSampling3D(size=pool_size)(conv6_2)
    up7_2 = Conv3D(64, (2, 2, 2), padding='same', activation='relu', kernel_initializer=tao_method)(up7_2)
    up7_2 = BatchNormalization()(up7_2)
    up7_2 = concatenate([up7_2, conv1_2], axis=4)
    conv7_2 = Conv3D(32, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(up7_2)
    conv7_2 = BatchNormalization()(conv7_2)
    conv7_2 = Conv3D(32, filter_size, padding='same', activation='relu', kernel_initializer=tao_method)(conv7_2)
    conv7_2 = BatchNormalization()(conv7_2)

    # conv8 = concatenate([conv7_1, conv7_2], axis=4)
    conv8 = Conv3D(n_labels, (1, 1, 1))(conv7_1)
    act = Activation('sigmoid')(conv8)
    model = Model(inputs=inputs, outputs=act)

    adam = optimizers.Adam()

    model.compile(optimizer=adam, loss=bwcl, metrics=[dice_coef, 'acc'])
    return model


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bwcl(y_true, y_pred):
    _epsilon = tf.convert_to_tensor(10e-8, y_pred.dtype.base_dtype)
    output = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    return K.mean(tf.nn.weighted_cross_entropy_with_logits(y_true, output, 10), axis=-1)




def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def tao_method(shape, dtype='float32'):
    data = InitData.__call__()
    print("initializing with size: " + str(shape))
    return tf.convert_to_tensor(np.array(data.get(shape[3], shape[4])), dtype=dtype)

def crop(start):
    def func(x):
            return x[:, :, :, :, start: start + 1]
    return Lambda(func)

