from __future__ import print_function
import Model as Model
import numpy as np
import os
from datetime import datetime
from matplotlib import pyplot as plt
from DS import DS

# initialization and prepare data set#########################################################
patch_size = [64, 64, 16]
batch_size = 16
epochs = 1
repeat = 5
K = 5
angles = [90, 180, 270]
scales = [0.5]
g_path = r'C:\result-' + datetime.now().strftime('%Y-%m-%d--%H-%M')

ds = DS('.\data\\', patch_size, K, angles, scales)
model = Model.get_model(input_shape=(patch_size[0], patch_size[1], patch_size[2], 1))

log = ds.create_files(K, repeat, g_path)

for fold in range(0,K):
    x_train, y_train, x_test, y_test = ds.get_data(fold)
    for repeat_count in range(0,repeat):
        path = g_path + r'\fold-' + str(fold) + r'rep-' + str(repeat_count)
        # train model##################################################################################
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(x_test, y_test))
        model.save_weights(path + r'\model.hd5')
        # get accuracy on test data####################################################################
        train_label_prediction = model.predict(x_train)
        ################################################model.evaluate(x_test, y_test)
        test_label_prediction = model.predict(x_test)
        # save images to file#######################################################################
        image = np.zeros([patch_size[0], patch_size[1], 3])

        for i in range(0, x_train.shape[0]):
            for j in range(0, patch_size[2]):
                image[:, :, 0] = x_train[i, :, :, j, 0] + (train_label_prediction[i, :, :, j, 0] / 4)  # red for predicted by model
                image[:, :, 1] = x_train[i, :, :, j, 0] + (y_train[i, :, :, j, 0]/4)  # green for ground truth
                image[:, :, 2] = x_train[i, :, :, j, 0]
                plt.imsave(path + r'\train' + str(fold+1) + '-' + str(i) + '-' + str(j) + '.png', image)

        for i in range(0, x_test.shape[0]):
            for j in range(0, patch_size[2]):
                image[:, :, 0] = x_test[i, :, :, j, 0] + (test_label_prediction[i, :, :, j, 0]/4)  # red for predicted by model
                image[:, :, 1] = x_test[i, :, :, j, 0] + (y_test[i, :, :, j, 0]/4)  # green for ground truth
                image[:, :, 2] = x_test[i, :, :, j, 0]
                plt.imsave(path + r'test' + str(fold+1) + '-' + str(i) + '-' + str(j) + '.png', image)