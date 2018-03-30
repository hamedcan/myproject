import numpy as np
import scipy.io

import matplotlib.pyplot as plt
from DS import DS
patch_size = [40, 40, 16]
batch_size = 16
epochs = 5
repeat = 1
channel = 2
K = 5
angles = []
scales = []
ds = DS('.\data\\', patch_size, channel, K, angles, scales)
x_train, y_train, x_test, y_test, x_test2, y_test2 = ds.get_data(1)


x = round(x_test.shape[1]/4)
y = round(x_test.shape[2]/4)
z = round(x_test.shape[3]/4)


x_tmp = scipy.ndimage.interpolation.zoom(x_test2[:, x:3*x, y:3*y, z:3*z, :], (1, 2, 2, 2, 1))
y_tmp = scipy.ndimage.interpolation.zoom(y_test2[:, x:3*x, y:3*y, z:3*z, :], (1, 2, 2, 2, 1))

for i in range(0,x_test.shape[0]):
    fig=plt.figure(figsize=(2, 2))

    fig.add_subplot(2, 2, 1)
    plt.imshow(x_test[i,:,:,5,0])

    fig.add_subplot(2, 2, 2)
    plt.imshow(y_test[i,:,:,5,0])

    fig.add_subplot(2, 2, 3)
    plt.imshow(x_tmp[i, :, :, 5, 0])

    fig.add_subplot(2, 2, 4)
    plt.imshow(np.around(y_tmp[i, :, :, 5, 0]))

    plt.show()