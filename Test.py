import numpy as np
import scipy.io
from xlrd import open_workbook
from matplotlib import pyplot as plt

label_maps = []
centers = []
count = 0
zc = 0
oc = 0
patch_size = [16, 16, 16]
path = '.\data\\'
wb = open_workbook(path + 'data.xlsx')
for s in wb.sheets():
    nrows = s._dimnrows
    file_name = [''] * nrows
    for i in range(nrows):
        centers.append([int(s.cell(i, 4).value), int(s.cell(i, 3).value), int(s.cell(i, 5).value)])
        file_name[i] = s.cell(i, 0).value
    count = nrows

for img_count in range(0, count):
    volname = file_name[img_count]
    center = centers[img_count]

    label_map = scipy.io.loadmat('.\data\gtruth_' + volname + '_fill.mat')
    label_map = label_map['gtruth_fill']
    label_map = np.reshape(label_map, (label_map.shape[0], label_map.shape[1], label_map.shape[2]))
    ystart = int(max([center[0] - patch_size[0] / 2, 0]))
    yend = int(min([center[0] + patch_size[0] / 2, label_map.shape[0]]))
    xstart = int(max([center[1] - patch_size[1] / 2, 0]))
    xend = int(min([center[1] + patch_size[1] / 2, label_map.shape[1]]))
    zstart = int(max([center[2] - patch_size[2] / 2, 0]))
    zend = int(min([center[2] + patch_size[2] / 2, label_map.shape[2]]))
    label_map = label_map[ystart:yend, xstart:xend, zstart:zend]

    # label_map_new = label_map.copy()
    # sizzzze = (int(yend-ystart), int(xend-xstart), int(zend-zstart))
    # label_map_new[ystart:yend, xstart:xend, zstart:zend] = np.zeros(sizzzze)
    #
    # rx = np.max(np.nonzero(label_map)[0])+ np.min(np.nonzero(label_map)[0])
    # ry = np.max(np.nonzero(label_map)[1])+ np.min(np.nonzero(label_map)[1])
    # rz = np.max(np.nonzero(label_map)[2])+ np.min(np.nonzero(label_map)[2])
    #
    #
    # print(str(np.min(np.nonzero(label_map)[0])),str(np.max(np.nonzero(label_map)[0])))
    # print(str(np.min(np.nonzero(label_map)[1])), str(np.max(np.nonzero(label_map)[1])))
    # print(str(np.min(np.nonzero(label_map)[2])), str(np.max(np.nonzero(label_map)[2])))
    # print(str(np.around(rx/2))+','+str(np.around(ry/2))+','+str(np.around(rz/2)))
    #
    # for i in range(0, label_map.shape[2]):
    #     if(np.count_nonzero(label_map_new[:, :, i])>0):
    #         print('i is ',str(i))
    #         plt.imshow(label_map_new[:, :, i])
    #         plt.show()
    #         plt.imshow(label_map[:, :, i])
    #         plt.show()
    oc = oc + (label_map == 1).sum()
    zc = zc + (label_map == 0).sum()

print(oc)
print(zc)
