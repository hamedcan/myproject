from __future__ import print_function

import numpy as np
import math
import scipy.io
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from xlrd import open_workbook


class DS:
    def __init__(self, path, patch_size, K=5, angles=[]):
        print('DS - initialization')
        self.path = path
        self.patch_size = patch_size
        self.K = K
        self.angles = angles

        self.kf = KFold(n_splits=K, shuffle=True)
        self.train_indexes = []
        self.test_indexes = []

        self.label_maps = []
        self.images = []
        self.centers = []
        self.count = 0

        self.load()
        self.augment()
        self.generate_k_fold_indexes()



    def load(self):
        print('DS - loading')
        wb = open_workbook(self.path + 'data.xlsx')
        for s in wb.sheets():
            nrows = s._dimnrows
            file_name = [''] * nrows
            center = np.zeros((nrows, 3))
            for i in range(nrows):
                file_name[i] = s.cell(i, 0).value
                center[i, 0] = s.cell(i, 4).value
                center[i, 1] = s.cell(i, 3).value
                center[i, 2] = s.cell(i, 5).value
            self.centers = np.array(center).astype(np.int)
            self.count = nrows

        for img_count in range(0, self.count):

            volname = file_name[img_count]

            label_map = scipy.io.loadmat('.\data\gtruth_' + volname + '_fill.mat')
            label_map = label_map['gtruth_fill']
            label_map = np.reshape(label_map, (label_map.shape[0], label_map.shape[1], label_map.shape[2]))
            self.label_maps.append(np.array(label_map))

            image = scipy.io.loadmat('.\data\enhanced_' + volname + '.mat')
            image = image['enhanced']
            image = np.reshape(image, (image.shape[0], image.shape[1], image.shape[3]))
            self.images.append(np.array(image))

    def generate_k_fold_indexes(self):
        for train_index, test_index in self.kf.split(self.images):
            self.train_indexes.append(train_index)
            self.test_indexes.append(test_index)

    def augment(self):
        print('DS - augmenting')
        for i in range(0, self.count):
            print('DS - image : ', i)
            for angle in self.angles:
                self.images.append(scipy.ndimage.rotate(self.images[i], angle=-angle))
                self.label_maps.append(scipy.ndimage.rotate(self.label_maps[i], angle=-angle))
                x, y = self.rotate(self.centers[i, 0], self.centers[i, 1], self.images[i].shape[0]/2, self.images[i].shape[1]/2, (angle/180)*math.pi)
                self.centers = np.vstack((self.centers, [x , y, self.centers[i,2]]))

                # print(self.centers[i])
                # print(self.centers[-1])
                # plt.imshow(self.images[-1][...,5])
                # plt.show()
                # plt.imshow(self.label_maps[-1][...,5])
                # plt.show()




    def rotate(self, px, py, ox, oy, angle):

        py_tmp = 2 * ox - px
        px_tmp = py

        px = px_tmp
        py = py_tmp

        px_tmp = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        py_tmp = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

        qy = 2 * ox - px_tmp
        qx = py_tmp
        return math.ceil(qx), math.ceil(qy)

    def get_data(self, fold):
        print('DS - start getting data')
        x_train = []
        y_train = []

        x_test = []
        y_test = []

        patch_size = self.patch_size

        train_count = len(self.train_indexes[fold])
        test_count = len(self.test_indexes[fold])
        # ======================================================================================================
        for i in self.train_indexes[fold]:
            image = self.images[i]
            label_map = self.label_maps[i]
            center = self.centers[i]
            ystart = max([center[0] - int(patch_size[0] / 2), 0])
            yend = min([center[0] + int(patch_size[0] / 2), image.shape[0]])
            xstart = max([center[1] - int(patch_size[1] / 2), 0])
            xend = min([center[1] + int(patch_size[1] / 2), image.shape[1]])
            zstart = max([center[2] - int(patch_size[2] / 2), 0])
            zend = min([center[2] + int(patch_size[2] / 2), image.shape[2]])

            image_tmp = np.zeros((int(patch_size[0]), int(patch_size[1]), int(patch_size[2])))
            label_map_tmp = np.zeros((int(patch_size[0]), int(patch_size[1]), int(patch_size[2])))
            real_size = [yend - ystart, xend - xstart, zend - zstart]

            image_tmp[:real_size[0], :real_size[1], :real_size[2]] = image[ystart:yend, xstart:xend, zstart:zend]
            x_train.append(image_tmp)

            label_map_tmp[:real_size[0], :real_size[1], :real_size[2]] = label_map[ystart:yend, xstart:xend,
                                                                         zstart:zend]
            y_train.append(label_map_tmp)

            # print(label_map_tmp.shape)
            # print(image_tmp.shape)
            # plt.imshow(image_tmp[...,5])
            # plt.show()
            # plt.imshow(label_map_tmp[...,5])
            # plt.show()

        x_train = np.reshape(np.array(x_train), (train_count, patch_size[0], patch_size[1], patch_size[2], 1))
        y_train = np.reshape(np.array(y_train), (train_count, patch_size[0], patch_size[1], patch_size[2], 1))

        # ======================================================================================================
        for i in self.test_indexes[fold]:
            image = self.images[i]
            label_map = self.label_maps[i]
            center = self.centers[i]
            ystart = max([center[0] - int(patch_size[0] / 2), 0])
            yend = min([center[0] + int(patch_size[0] / 2), image.shape[0]])
            xstart = max([center[1] - int(patch_size[1] / 2), 0])
            xend = min([center[1] + int(patch_size[1] / 2), image.shape[1]])
            zstart = max([center[2] - int(patch_size[2] / 2), 0])
            zend = min([center[2] + int(patch_size[2] / 2), image.shape[2]])

            image_tmp = np.zeros((int(patch_size[0]), int(patch_size[1]), int(patch_size[2])))
            label_map_tmp = np.zeros((int(patch_size[0]), int(patch_size[1]), int(patch_size[2])))
            real_size = [yend - ystart, xend - xstart, zend - zstart]

            image_tmp[:real_size[0], :real_size[1], :real_size[2]] = image[ystart:yend, xstart:xend, zstart:zend]
            x_test.append(image_tmp)

            label_map_tmp[:real_size[0], :real_size[1], :real_size[2]] = label_map[ystart:yend, xstart:xend,
                                                                             zstart:zend]
            y_test.append(label_map_tmp)

            # print(label_map_tmp.shape)
            # print(image_tmp.shape)
            # plt.imshow(image_tmp[...,5])
            # plt.show()
            # plt.imshow(label_map_tmp[...,5])
            # plt.show()

        x_test = np.reshape(np.array(x_test), (test_count, patch_size[0], patch_size[1], patch_size[2], 1))
        y_test = np.reshape(np.array(y_test), (test_count, patch_size[0], patch_size[1], patch_size[2], 1))

        return x_train, y_train, x_test, y_test
