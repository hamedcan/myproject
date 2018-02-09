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
            # nz = np.nonzero(self.label_maps[i])
            # x = np.max(nz[0]) - np.min(nz[0])
            # y = np.max(nz[1]) - np.min(nz[1])
            # z = np.max(nz[2]) - np.min(nz[2])

            # if x <= self.patch_size[0] and y <= self.patch_size[1] and z <= self.patch_size[2]:
            train_image.append(self.images[i])
            train_label_map.append(self.label_maps[i])
            train_center.append([self.centers[i][0], self.centers[i][1], self.centers[i][2]])
            # else:
            #     train_image.append(scipy.ndimage.interpolation.zoom(self.images[i], 0.5))
            #     train_label_map.append(np.around(scipy.ndimage.interpolation.zoom(self.label_maps[i], 0.5)))
            #     train_center.append([round(self.centers[i][0]*0.5), round(self.centers[i][1]*0.5), round(self.centers[i][2]*0.5)])




        # ================================zoom out=========================================
        train_count = self.augment_zoom(train_count, train_image, train_label_map, train_center, self.scales)
        # ================================rotation=========================================
        train_count = self.augment_rotation(train_count, train_image, train_label_map, train_center)
        # ==================================flip===========================================
        train_count = self.augment_flip(train_count, train_image, train_label_map, train_center)
        # ======================================================================================================
        for i in range(0, train_count):
            image = train_image[i]
            label_map = train_label_map[i]
            center = train_center[i]
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

        x_train = np.reshape(np.array(x_train), (train_count, patch_size[0], patch_size[1], patch_size[2], 1))
        y_train = np.reshape(np.array(y_train), (train_count, patch_size[0], patch_size[1], patch_size[2], 1))

        # ===============t================e======================s=================t================================
        for i in self.test_indexes[fold]:
            # nz = np.nonzero(self.label_maps[i])
            # x = np.max(nz[0]) - np.min(nz[0])
            # y = np.max(nz[1]) - np.min(nz[1])
            # z = np.max(nz[2]) - np.min(nz[2])

            # if x <= self.patch_size[0] and y <= self.patch_size[1] and z <= self.patch_size[2]:
            image = self.images[i]
            label_map = self.label_maps[i]
            center = self.centers[i]
            # else:
            #     image = scipy.ndimage.interpolation.zoom(self.images[i], 0.5)
            #     label_map = np.around(scipy.ndimage.interpolation.zoom(self.label_maps[i], 0.5))
            #     center = [round(self.centers[i][0]*0.5), round(self.centers[i][1]*0.5), round(self.centers[i][2]*0.5)]

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

        x_test = np.reshape(np.array(x_test), (test_count, patch_size[0], patch_size[1], patch_size[2], 1))
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
