import numpy as np
import math
import scipy.io
import scipy.ndimage
import matplotlib.pyplot as plt
from xlrd import open_workbook
patch_size = [64, 64, 16]
batch_size = 2
epochs = 1000
max_gray_level = 4096
testrate = 0
K = 5
angles = [90, 45]
path = '.\data\\'

train_indexes = []
test_indexes = []

label_maps = []
images = []
centers = []
count = 0

wb = open_workbook(path + 'data.xlsx')
for s in wb.sheets():
    nrows = s._dimnrows
    file_name = [''] * nrows
    center = np.zeros((nrows, 3))
    for i in range(nrows):
        file_name[i] = s.cell(i, 0).value
        center[i, 0] = s.cell(i, 4).value
        center[i, 1] = s.cell(i, 3).value
        center[i, 2] = s.cell(i, 5).value
    centers = np.array(center).astype(np.int)
    count = nrows

for img_count in range(0, 5):
    volname = file_name[img_count]

    label_map = scipy.io.loadmat('.\data\gtruth_' + volname + '_fill.mat')
    label_map = label_map['gtruth_fill']
    label_map = np.reshape(label_map, (label_map.shape[0], label_map.shape[1], label_map.shape[2]))
    label_maps.append(np.array(label_map))

    image = scipy.io.loadmat('.\data\enhanced_' + volname + '.mat')
    image = image['enhanced']
    image = np.reshape(image, (image.shape[0], image.shape[1], image.shape[3]))
    images.append(np.array(image))


i = 4
print(centers[i])
plt.imshow(images[i][...,20])
plt.show()
plt.imshow(label_maps[i][...,20])
plt.show()

hamed = images[i][:,::-1,:]
shapan = label_maps[i][:,::-1,:]

plt.imshow(hamed[...,20])
plt.show()
plt.imshow(shapan[...,20])
plt.show()





