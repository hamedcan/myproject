from __future__ import print_function
import Model
import numpy as np
from matplotlib import pyplot as plt
from DS import DS

# initialization and prepare data set#########################################################
patch_size = [64, 64, 16]
batch_size = 5
epochs = 1000
max_gray_level = 4096
testrate = 0
K = 5
angles = [90, 180, 270]
ds = DS('.\data\\', patch_size, K, angles)

for fold in range(0,K):
    print('===================================================================================================',K+1)
    x_train, y_train, x_test, y_test = ds.get_data(fold)
    # define mode##################################################################################
    model = Model.get_model(input_shape=(patch_size[0], patch_size[1], patch_size[2], 1))
    # train model##################################################################################
    print('epoch: ', epochs)
    print('batch size: ', batch_size)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(x_test, y_test))
    # get accuracy on test data####################################################################
    classes = model.predict(x_train)
    print(model.evaluate(x_test, y_test))

    # save images to file#######################################################################
    image = np.zeros([patch_size[0], patch_size[1], 3])
    for i in range(0, x_train.shape[0]):
        for j in range(0, patch_size[2]):
            image[:, :, 0] = classes[i, :, :, j, 0]  # red for predicted by model
            image[:, :, 1] = y_train[i, :, :, j, 0]  # green for ground truth
            plt.imsave('D:\\result\\' + str(fold+1) + '-' + str(i) + '-' + str(j) + '.jpg', image)
