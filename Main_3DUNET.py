from __future__ import print_function
import Model as Model
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from DS import DS

# initialization and prepare data set#########################################################
patch_size = [64, 64, 16]
batch_size = 16
epochs = 2
repeat = 5
K = 5
angles = []
scales = []
g_path = r'C:\result\\' + datetime.now().strftime('%Y-%m-%d--%H-%M')

ds = DS('.\data\\', patch_size, K, angles, scales)
logger = ds.create_files(K, repeat, g_path)
logger.write('batch size: ' + str(batch_size) + '\n')
logger.write('epochs: ' + str(epochs) + '\n')
logger.write('repeat: ' + str(repeat) + '\n')
logger.write('fold: ' + str(K) + '\n')
logger.write('angles: ' + str(angles) + '\n')
logger.write('scales: ' + str(scales) + '\n')
logger.flush()

for fold in range(0,K):
    x_train, y_train, x_test, y_test = ds.get_data(fold)
    logger.write('===================fold: ' + str(fold) + '===================\n')
    print('===================fold: ' + str(fold) + '===================\n')
    for repeat_count in range(0,repeat):
        disable_count = fold + repeat_count
        model = Model.get_model(logger ,disable_count ,input_shape=(patch_size[0], patch_size[1], patch_size[2], 1))
        logger.write('repeat: ' + str(repeat_count) + '\n')
        print('repeat: ' + str(repeat_count) + '\n')
        path = g_path + r'\fold-' + str(fold) + r'-rep-' + str(repeat_count)
        # train model##################################################################################
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(x_test, y_test))
        model.save_weights(path + r'\model.hd5')
        # get accuracy on test data####################################################################
        train_label_prediction = model.predict(x_train)
        test_label_prediction = model.predict(x_test)

        logger.write('train accuracy: ' + str(model.evaluate(x_train, y_train)[1]) + '\n')
        logger.write('test accuracy: ' + str(model.evaluate(x_test, y_test)[1]) + '\n\n')
        # save images to file#######################################################################
        image = np.zeros([patch_size[0], patch_size[1], 3])
        logger.flush()
        for i in range(0, x_train.shape[0]):
            for j in range(0, patch_size[2]):
                image[:, :, 0] = x_train[i, :, :, j, 0] + (train_label_prediction[i, :, :, j, 0] / 4)  # red for predicted by model
                image[:, :, 1] = x_train[i, :, :, j, 0] + (y_train[i, :, :, j, 0]/4)  # green for ground truth
                image[:, :, 2] = x_train[i, :, :, j, 0]
                plt.imsave(path + r'\train' + '\im-' + str(i) + '-' + str(j) + '.png', image)

        for i in range(0, x_test.shape[0]):
            for j in range(0, patch_size[2]):
                image[:, :, 0] = x_test[i, :, :, j, 0] + (test_label_prediction[i, :, :, j, 0]/4)  # red for predicted by model
                image[:, :, 1] = x_test[i, :, :, j, 0] + (y_test[i, :, :, j, 0]/4)  # green for ground truth
                image[:, :, 2] = x_test[i, :, :, j, 0]
                plt.imsave(path + r'\test' + '\im-' + str(i) + '-' + str(j) + '.png', image)

logger.close()