from __future__ import print_function
import Model_3Dunet as Model
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from DS import DS
from keras import backend as Keras

# initialization and prepare data set#########################################################
for rcount in range(0, 1):
    patch_size = [32, 32, 16]
    batch_size = 32
    epochs = 2
    repeat = 1
    channel = 1
    K = 2
    angles = []
    scales = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    # scales = [0.8, 0.6, 0.4]
    g_path = r'C:\result\\' + datetime.now().strftime('%Y-%m-%d--%H-%M')

    ds = DS('.\data\\', patch_size, channel, K, angles, scales)
    logger = ds.create_files(K, repeat, g_path)
    model = Model.get_model(logger, 0, input_shape=(patch_size[0], patch_size[1], patch_size[2], channel))
    logger.write('batch size: ' + str(batch_size) + '\n')
    logger.write('epochs: ' + str(epochs) + '\n')
    logger.write('repeat: ' + str(repeat) + '\n')
    logger.write('fold: ' + str(K) + '\n')
    logger.write('angles: ' + str(angles) + '\n')
    logger.write('scales: ' + str(scales) + '\n')
    logger.write('channel: ' + str(channel) + '\n')
    logger.flush()
    ####################################################################################################
    for fold in range(0, K):
        x_train, y_train, x_test, y_test = ds.get_data(fold)
        logger.write('===================fold: ' + str(fold) + '===================\n')
        logger.write('contain:' + str(ds.train_indexes[fold]) + '\n\n')
        print('===================fold: ' + str(fold) + '===================\n')
        for repeat_count in range(0, repeat):
            del model
            Keras.clear_session()
            model = Model.get_model(logger, 1, input_shape=(patch_size[0], patch_size[1], patch_size[2], channel))
            logger.write('repeat: ' + str(repeat_count) + '\n')
            print('repeat: ' + str(repeat_count) + '\n')
            path = g_path + r'\fold-' + str(fold) + r'-rep-' + str(repeat_count)
            # train model##################################################################################
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                      validation_data=(x_test[0], y_test[0]), verbose=2)
            # start test##################################################################################
            ds.post_process2(fold, logger, x_test, y_test, model)
            # end test##################################################################################
            logger.flush()
            logger.close()
