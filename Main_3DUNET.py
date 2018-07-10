from __future__ import print_function
import Model_3Dunet as Model
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from DS import DS
from keras import backend as Keras
from Init_data import InitData
import scipy.io

# initialization and prepare data set#########################################################
patch_size = [40, 40, 16]
batch_size = 16
epochs = 200
repeat = 5
channel = 2
K = 5
angles = []
scales = [0.75,0.5,0.25]
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

# pre train model##################################################################################


####################################################################################################
for fold in range(0, K):
    x_train, y_train, x_test, y_test, x_train3, y_train3, x_test3, y_test3, x_train4, y_train4, x_test4, y_test4 = ds.get_data(fold)
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
                  validation_data=(x_test, y_test), verbose=2)
        model.save_weights(path + r'\model.hd5')
        # get accuracy on test data####################################################################
        pred = model.predict(x_test)
        ds.post_process2(fold, logger, y_test, pred, '1X')

        pred = model.predict(x_test3)
        ds.post_process2(fold, logger, y_test3, pred, '3X')

        pred = model.predict(x_test4)
        ds.post_process2(fold, logger, y_test4, pred, '4X')

# logging#####################################################################################
        logger.write('==========================================\n')
        # save images to file##################################################################
        logger.flush()
        #
        # for i in range(0, x_test.shape[0]):
        #     for j in range(0, patch_size[2]):
        #
        #         image = np.zeros([patch_size[0], patch_size[1], 3])
        #         image[:, :, 0] = x_test[i, :, :, j, 0] + (pred[i, :, :, j, 0]/4)  # red for predicted by model
        #         image[:, :, 1] = x_test[i, :, :, j, 0]
        #         image[:, :, 2] = x_test[i, :, :, j, 0]
        #         plt.imsave(path + r'\test' + '\im-' + str(i) + '-' + str(j) + '-pred.png', image)
        #
        #         image = np.zeros([patch_size[0], patch_size[1], 3])
        #         image[:, :, 0] = x_test[i, :, :, j, 0]
        #         image[:, :, 1] = x_test[i, :, :, j, 0] + (y_test[i, :, :, j, 0]/4)  # green for ground truth
        #         image[:, :, 2] = x_test[i, :, :, j, 0]
        #         plt.imsave(path + r'\test' + '\im-' + str(i) + '-' + str(j) + '-gt.png', image)
        #
        #         image = np.zeros([patch_size[0], patch_size[1], 3])
        #         image[:, :, 0] = x_test[i, :, :, j, 0]
        #         image[:, :, 1] = x_test[i, :, :, j, 0]
        #         image[:, :, 2] = x_test[i, :, :, j, 0]
        #         plt.imsave(path + r'\test' + '\im-' + str(i) + '-' + str(j) + '-orig.png', image)

logger.close()
