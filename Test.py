from __future__ import print_function
import matplotlib.pylab as plt
import numpy as np
import scipy.io
import scipy.ndimage

image_final = np.zeros((300, 300, 300, 3))
center = [300, 300, 2]
image = plt.imread("D:\\Pictures\\wallpaper\\bear.jpg")
print(image.shape)
scales = [(1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)]
for i in (0, 1, 2):
    image_tmp = scipy.ndimage.interpolation.zoom(image, scales[i])

    ystart = int(max([center[0]*scales[i][0] - 300 / 2, 0]))
    yend = int(min([center[0]*scales[i][0] + 300 / 2, image_tmp.shape[0]]))
    xstart = int(max([center[1]*scales[i][1] - 300 / 2, 0]))
    xend = int(min([center[1]*scales[i][1] + 300 / 2, image_tmp.shape[1]]))
    zstart = int(max([center[2]*scales[i][2] - 3 / 2, 0]))
    zend = int(min([center[2]*scales[i][2] + 3 / 2, image_tmp.shape[2]]))

    image_tmp = image_tmp[ystart:yend, xstart:xend, zstart:zend]

    ystart = int((300 - image_tmp.shape[0])/2)
    yend = int(ystart + image_tmp.shape[0])
    xstart = int((300 - image_tmp.shape[1])/2)
    xend = int(xstart + image_tmp.shape[1])
    zstart = int((300 - image_tmp.shape[2])/2)
    zend = int(zstart + image_tmp.shape[2])
    # image_final[ystart:yend, xstart:xend, zstart:zend, i] = image_tmp
    plt.imshow(image_tmp[:,:,0])
    plt.show()
