from __future__ import print_function

import numpy as np
import math
import scipy.io
import os
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from xlrd import open_workbook


class DS:
    def __init__(self, path, patch_size, K=5, angles=[], scales=[]):
        print('DS - initialization')
        self.path = path
        self.patch_size = patch_size
        self.K = K
        self.angles = angles
        self.scales = scales

        self.kf = KFold(n_splits=K, shuffle=True)
        self.train_indexes = []
        self.test_indexes = []

        self.label_maps = []
        self.images = []
        self.centers = []
        self.count = 0

        self.load()
        self.generate_k_fold_indexes()

    def load(self):
        print('DS - loading')
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

    def augment_zoom(self, count, images, label_maps, centers, scale):
        for j in scale:
            print('DS - augmenting_zoom by scale: ', str(j))
            for i in range(0, count):
                # print('DS - zoom - image : ', i)
                images.append(scipy.ndimage.interpolation.zoom(images[i], j))
                label_maps.append(np.around(scipy.ndimage.interpolation.zoom(label_maps[i], j)))
                centers.append([round(centers[i][0]*j), round(centers[i][1]*j), round(centers[i][2]*j)])
        return count * (len(scale) + 1)

    def augment_rotation(self, count, images, label_maps, centers):
        print('DS - augmenting__rotation')
        for i in range(0, count):
            print('DS - rotation - image : ', i)
            for angle in self.angles:
                images.append(scipy.ndimage.rotate(images[i], angle=-angle))
                label_maps.append(scipy.ndimage.rotate(label_maps[i], angle=-angle))
                x, y = self.rotate(centers[i][0], centers[i][1], images[i].shape[0]/2, images[i].shape[1]/2, (-angle/180)*math.pi)
                centers.append([x, y, centers[i][2]])
        return count * (len(self.angles) + 1)

    def augment_flip(self, count, images, label_maps, centers):
        print('DS - augmenting__fliping')
        for i in range(0, count):
            # print('DS - fliping - image : ', i)
            images.append(images[i][:, ::-1, :])
            label_maps.append(label_maps[i][:, ::-1, :])
            x, y = [centers[i][0], images[i].shape[1] - centers[i][1]]
            centers.append([x, y, centers[i][2]])
        return count * 2

    def rotate(self, px, py, ox, oy, angle):
        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return math.ceil(qx), math.ceil(qy)

    def get_data(self, fold):
        print('DS - start getting data')
        train_count = len(self.train_indexes[fold])
        test_count = len(self.test_indexes[fold])

        x_train = []
        y_train = []

        x_test = []
        y_test = []

        train_image = []
        train_label_map = []
        train_center = []

        patch_size = self.patch_size

        for i in self.train_indexes[fold]:
            train_image.append(self.images[i])
            train_label_map.append(self.label_maps[i])
            train_center.append([self.centers[i][0], self.centers[i][1], self.centers[i][2]])
        # ================================zoom out=========================================
        train_count = self.augment_zoom(train_count, train_image, train_label_map, train_center, self.scales)
        # ================================rotation=========================================
        train_count = self.augment_rotation(train_count, train_image, train_label_map, train_center)
        # ==================================flip===========================================
        train_count = self.augment_flip(train_count, train_image, train_label_map, train_center)
        # ======================================================================================================
        for i in range(0, train_count):
            self.add_8(train_image[i], train_label_map[i], train_center[i], x_train, y_train)
        x_train = np.reshape(np.array(x_train), (train_count, patch_size[0], patch_size[1], patch_size[2], 3))
        y_train = np.reshape(np.array(y_train), (train_count, patch_size[0], patch_size[1], patch_size[2], 1))

        # ===============t================e======================s=================t================================
        for i in self.test_indexes[fold]:
            self.add_8(self.images[i], self.label_maps[i], self.centers[i], x_test, y_test)
        x_test = np.reshape(np.array(x_test), (test_count, patch_size[0], patch_size[1], patch_size[2], 3))
        y_test = np.reshape(np.array(y_test), (test_count, patch_size[0], patch_size[1], patch_size[2], 1))

        return x_train, y_train, x_test, y_test

    @staticmethod
    def create_files(K, R, g_path):
        for k in range(0, K):
            for r in range(0, R):
                path = g_path + r'\fold-' + str(k) + r'-rep-' + str(r)
                if not os.path.exists(path):
                    os.makedirs(path)

                if not os.path.exists(path + r'\train'):
                    os.makedirs(path + r'\train')

                if not os.path.exists(path + r'\test'):
                    os.makedirs(path + r'\test')

        open(g_path + '\info.txt', "w+").close()
        return open(g_path + '\info.txt', "a")

    def add(self, image, label_map, center, x, y):
        patch_size = self.patch_size
        image_final = np.zeros((int(patch_size[0]), int(patch_size[1]), int(patch_size[2]), 3))
        label_map_final = np.zeros((int(patch_size[0]), int(patch_size[1]), int(patch_size[2]), 1))

        scales = [(1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)]
        for i in (0, 1, 2):

            image_tmp = scipy.ndimage.interpolation.zoom(image, scales[i])
            if i == 0:
                label_map_tmp = np.around(scipy.ndimage.interpolation.zoom(label_map, scales[i]))

            ystart = int(max([center[0]*scales[i] - patch_size[0] / 2, 0]))
            yend = int(min([center[0]*scales[i] + patch_size[0] / 2, image_tmp.shape[0]]))
            xstart = int(max([center[1]*scales[i] - patch_size[1] / 2, 0]))
            xend = int(min([center[1]*scales[i] + patch_size[1] / 2, image_tmp.shape[1]]))
            zstart = int(max([center[2]*scales[i] - patch_size[2] / 2, 0]))
            zend = int(min([center[2]*scales[i] + patch_size[2] / 2, image_tmp.shape[2]]))

            image_tmp = image_tmp[ystart:yend, xstart:xend, zstart:zend]
            if i == 0:
                label_map_tmp = label_map_tmp[ystart:yend, xstart:xend,zstart:zend]

            ystart = int((patch_size[0] - image_tmp.shape[0])/2)
            yend = int(ystart + image_tmp.shape[0])
            xstart = int((patch_size[1] - image_tmp.shape[1])/2)
            xend = int(xstart + image_tmp.shape[1])
            zstart = int((patch_size[2] - image_tmp.shape[2])/2)
            zend = int(zstart + image_tmp.shape[2])

            image_final[ystart:yend, xstart:xend, zstart:zend, i] = image_tmp
            if i == 0:
                label_map_final[ystart:yend, xstart:xend, zstart:zend, i] = label_map_tmp

        x.append(image_final)
        y.append(label_map_final)

    def add_8(self, image, label_map, center, x, y):
        tmp_center = [0, 0, 0]
        patch_size = self.patch_size
        for p in (-patch_size[0]/2, patch_size[0]/2):
            for q in (-patch_size[1]/2, patch_size[1]/2):
                for r in (-patch_size[2]/2, patch_size[2]/2):
                    tmp_center[0] = center[0] + p
                    tmp_center[1] = center[1] + q
                    tmp_center[2] = center[2] + r
                    if all(tmp > 0 for tmp in tmp_center):
                        self.add(image, label_map, tmp_center, x, y)

