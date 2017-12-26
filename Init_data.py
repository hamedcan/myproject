import random

import numpy as np
import math
import scipy.io
from xlrd import open_workbook


class Singleton(type):
    _instances = {}

    def __call__(cls):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__()
        return cls._instances[cls]


class InitData(metaclass=Singleton):
    path = '.\data\\'
    centers = []
    count = 0
    data = []

    def __init__(self):
        size = 1
        print('Weight Initialize - loading')
        wb = open_workbook(self.path + 'data.xlsx')
        for s in wb.sheets():
            nrows = s._dimnrows
            file_name = [''] * nrows
            for i in range(nrows):
                self.centers.append([int(s.cell(i, 4).value), int(s.cell(i, 3).value), int(s.cell(i, 5).value)])
                file_name[i] = s.cell(i, 0).value
            self.count = nrows

        for img_count in range(0, self.count):
            volname = file_name[img_count]

            label_map = scipy.io.loadmat('.\data\gtruth_' + volname + '_perim.mat')
            label_map = label_map['gtruth_perim']
            label_map = np.reshape(label_map, (label_map.shape[0], label_map.shape[1], label_map.shape[2]))

            image = scipy.io.loadmat('.\data\enhanced_' + volname + '.mat')
            image = image['enhanced']
            image = np.reshape(image, (image.shape[0], image.shape[1], image.shape[3]))

            nz = np.nonzero(label_map)

            for voxel_number in range(0, np.count_nonzero(label_map) - 1):
                v_idx_x = nz[0][voxel_number]
                v_idx_y = nz[1][voxel_number]
                v_idx_z = nz[2][voxel_number]
                self.data.append(image[v_idx_x - size:v_idx_x + size + 1, v_idx_y - size:v_idx_y + size + 1,
                                 v_idx_z - size:v_idx_z + size + 1])

    def get(self, channel_count, filter_count):
        result = np.zeros((3, 3, 3, channel_count, filter_count));
        for i in range(0, channel_count):
            for j in range(0, filter_count):
                print(str(i) + "---" + str(j))
                result[:, :, :, i, j] = self.data[random.randint(0, len(self.data) - 1)]

        return result
