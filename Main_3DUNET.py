from __future__ import print_function
import Model_3Dunet as Model
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from DS import DS
from Init_data import InitData
import scipy.io

# initialization and prepare data set#########################################################
patch_size = [40, 40, 16]
batch_size = 16
epochs = 200
repeat = 3
channel = 2
K = 5
angles = []
scales = [0.5]
g_path = r'C:\result\\' + datetime.now().strftime('%Y-%m-%d--%H-%M')

ds = DS('.\data\\', patch_size, channel, K, angles, scales)
logger = ds.create_files(K, repeat, g_path)
model = Model.get_model(logger, 0, input_shape=(patch_size[0], patch_size[1], patch_size[2], 1))
logger.write('batch size: ' + str(batch_size) + '\n')
logger.write('epochs: ' + str(epochs) + '\n')
logger.write('repeat: ' + str(repeat) + '\n')
logger.write('fold: ' + str(K) + '\n')
logger.write('angles: ' + str(angles) + '\n')
logger.write('scales: ' + str(scales) + '\n')
logger.write('channel: ' + str(channel) + '\n')
logger.flush()

# init_data = InitData.__call__()

for fold in range(0, K):
    x_train, y_train, x_test, y_test, x_train2, y_train2, x_test2, y_test2 = ds.get_data(fold)
    logger.write('===================fold: ' + str(fold) + '===================\n')
    logger.write('contain:' + str(ds.train_indexes[fold]) + '\n\n')
    print('===================fold: ' + str(fold) + '===================\n')
    for repeat_count in range(0, repeat):
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
        pred2 = model.predict(x_test2)
        micro, macro = DS.post_process(logger, y_test, pred, y_test2, pred2)
        x = round(x_test.shape[1] / 4)
        y = round(x_test.shape[2] / 4)
        z = round(x_test.shape[3] / 4)
        x_tmp = scipy.ndimage.interpolation.zoom(x_test2[:, x:3 * x, y:3 * y, z:3 * z, :], (1, 2, 2, 2, 1))
        y_tmp = scipy.ndimage.interpolation.zoom(y_test2[:, x:3 * x, y:3 * y, z:3 * z, :], (1, 2, 2, 2, 1))
        # logging#######################################################################################
        logger.write('==========================================\n')
        logger.write('train accuracy:\t' + str(model.evaluate(x_train, y_train)[1]) + '\n')
        logger.write('test accuracy: \t' + str(model.evaluate(x_test, y_test)[1]) + '\n\n')

        logger.write('train2 accuracy:\t' + str(model.evaluate(x_train2, y_train2)[1]) + '\n')
        logger.write('test2 accuracy: \t' + str(model.evaluate(x_test2, y_test2)[1]) + '\n\n')

        logger.write('test 0.5*2 accuracy: \t' + str(model.evaluate(x_tmp, y_tmp)[1]) + '\n')
        logger.write('my method: ' + str(micro) + '  ' + str(macro) + '\n')

        # save images to file#######################################################################
        logger.flush()

        for i in range(0, x_test.shape[0]):
            for j in range(0, patch_size[2]):
                # image[:, :, 0] = x_test[i, :, :, j, 0] + (pred[i, :, :, j, 0]/4)  # red for predicted by model
                # image[:, :, 1] = x_test[i, :, :, j, 0] + (y_test[i, :, :, j, 0]/4)  # green for ground truth
                # image[:, :, 2] = x_test[i, :, :, j, 0]
                # plt.imsave(path + r'\test' + '\im-' + str(i) + '-' + str(j) + '.png', image)


                # image[:, :, 0] = x_test[i, :, :, j, 0] + (pred[i, :, :, j, 0]/4)  # red for predicted by model
                # image[:, :, 1] = x_test[i, :, :, j, 0]
                # image[:, :, 2] = x_test[i, :, :, j, 0]
                # plt.imsave(path + r'\test' + '\im-' + str(i) + '-' + str(j) + '-p.png', image)
                #
                # image[:, :, 0] = x_test[i, :, :, j, 0]
                # image[:, :, 1] = x_test[i, :, :, j, 0] + (y_test[i, :, :, j, 0]/4)  # green for ground truth
                # image[:, :, 2] = x_test[i, :, :, j, 0]
                # plt.imsave(path + r'\test' + '\im-' + str(i) + '-' + str(j) + '-gt.png', image)
                #
                # image[:, :, 0] = x_test[i, :, :, j, 0]
                # image[:, :, 1] = x_test[i, :, :, j, 0]
                # image[:, :, 2] = x_test[i, :, :, j, 0]
                # plt.imsave(path + r'\test' + '\im-' + str(i) + '-' + str(j) + '-orig.png', image)


                image = np.zeros([patch_size[0], patch_size[1], 3])
                image[:, :, 0] = (pred[i, :, :, j, 0])  # red for predicted by model
                plt.imsave(path + r'\test' + '\im-' + str(i) + '-' + str(j) + '-p.png', image)

                image = np.zeros([patch_size[0], patch_size[1], 3])
                image[:, :, 1] = (y_test[i, :, :, j, 0])  # green for ground truth
                plt.imsave(path + r'\test' + '\im-' + str(i) + '-' + str(j) + '-gt.png', image)

logger.close()
