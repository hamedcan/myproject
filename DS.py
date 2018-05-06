from __future__ import print_function

import numpy as np
import math
import scipy.io
import os
from sklearn.model_selection import KFold
from xlrd import open_workbook


class DS:
    def __init__(self, path, patch_size, K=5):
        print('DS - initialization')
        self.path = path
        self.patch_size = patch_size
        self.K = K

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

    def get_data(self, fold):
        print('DS - start getting data')
        train_count = len(self.train_indexes[fold])
        test_count = len(self.test_indexes[fold])

        x_train = []
        y_train = []

        x_train2 = []
        y_train2 = []

        x_train3 = []
        y_train3 = []

        x_test = []
        y_test = []

        x_test2 = []
        y_test2 = []

        x_test3 = []
        y_test3 = []

        scales = [0.5, 0.25]

        train_image = []
        train_label_map = []
        train_center = []

        patch_size = self.patch_size

        for i in self.train_indexes[fold]:
            train_image.append(self.images[i])
            train_label_map.append(self.label_maps[i])
            train_center.append([self.centers[i][0], self.centers[i][1], self.centers[i][2]])
        for i in range(0, train_count):
            self.add(train_image[i], train_label_map[i], train_center[i], x_train, y_train)

            self.add(scipy.ndimage.interpolation.zoom(train_image[i], scales[0]),
                     np.around(scipy.ndimage.interpolation.zoom(train_label_map[i], scales[0])),
                     [round(train_center[i][0] * scales[0]), round(train_center[i][1] * scales[0]),
                      round(train_center[i][2] * scales[0])], x_train2, y_train2)

            self.add(scipy.ndimage.interpolation.zoom(train_image[i], scales[1]),
                     np.around(scipy.ndimage.interpolation.zoom(train_label_map[i], scales[1])),
                     [round(train_center[i][0] * scales[1]), round(train_center[i][1] * scales[1]),
                      round(train_center[i][2] * scales[1])], x_train3, y_train3)

        x_train = np.reshape(np.array(x_train),
                             (len(x_train), patch_size[0], patch_size[1], patch_size[2], self.channel))
        y_train = np.reshape(np.array(y_train), (len(x_train), patch_size[0], patch_size[1], patch_size[2], 1))

        x_train2 = np.reshape(np.array(x_train2),
                             (len(x_train2), patch_size[0], patch_size[1], patch_size[2], self.channel))
        y_train2 = np.reshape(np.array(y_train2), (len(x_train2), patch_size[0], patch_size[1], patch_size[2], 1))

        x_train3 = np.reshape(np.array(x_train3),
                             (len(x_train2), patch_size[0], patch_size[1], patch_size[2], self.channel))
        y_train3 = np.reshape(np.array(y_train3), (len(x_train3), patch_size[0], patch_size[1], patch_size[2], 1))
        for i in self.test_indexes[fold]:
            self.add(self.images[i], self.label_maps[i], self.centers[i], x_test, y_test)
            self.add(scipy.ndimage.interpolation.zoom(self.images[i], scales[0]),
                     np.around(scipy.ndimage.interpolation.zoom(self.label_maps[i], scales[0])),
                     [round(self.centers[i][0] * scales[0]), round(self.centers[i][1] * scales[0]),
                      round(self.centers[i][2] * scales[0])], x_test2, y_test2)

            self.add(scipy.ndimage.interpolation.zoom(self.images[i], scales[1]),
                     np.around(scipy.ndimage.interpolation.zoom(self.label_maps[i], scales[1])),
                     [round(self.centers[i][0] * scales[1]), round(self.centers[i][1] * scales[1]),
                      round(self.centers[i][2] * scales[1])], x_test3, y_test3)


        x_test = np.reshape(np.array(x_test), (len(x_test), patch_size[0], patch_size[1], patch_size[2], self.channel))
        y_test = np.reshape(np.array(y_test), (len(x_test), patch_size[0], patch_size[1], patch_size[2], 1))

        x_test2 = np.reshape(np.array(x_test2),
                             (len(x_test2), patch_size[0], patch_size[1], patch_size[2], self.channel))
        y_test2 = np.reshape(np.array(y_test2), (len(x_test2), patch_size[0], patch_size[1], patch_size[2], 1))

        x_test3 = np.reshape(np.array(x_test3),
                             (len(x_test2), patch_size[0], patch_size[1], patch_size[2], self.channel))
        y_test3 = np.reshape(np.array(y_test3), (len(x_test3), patch_size[0], patch_size[1], patch_size[2], 1))

        return x_train, y_train, x_test, y_test, x_train2, y_train2, x_test2, y_test2, x_train3, y_train3, x_test3, y_test3

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
        image_final = np.ones((int(patch_size[0]), int(patch_size[1]), int(patch_size[2]), self.channel))
        label_map_final = np.zeros((int(patch_size[0]), int(patch_size[1]), int(patch_size[2])))

        scales = [(1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)]
        for i in range(0, self.channel):

            if scales[i] == (1, 1, 1):
                image_tmp = image
            else:
                image_tmp = scipy.ndimage.interpolation.zoom(image, scales[i])

            if i == 0:
                label_map_tmp = label_map

            ystart = int(max([center[0] * scales[i][0] - patch_size[0] / 2, 0]))
            yend = int(min([center[0] * scales[i][0] + patch_size[0] / 2, image_tmp.shape[0]]))
            xstart = int(max([center[1] * scales[i][1] - patch_size[1] / 2, 0]))
            xend = int(min([center[1] * scales[i][1] + patch_size[1] / 2, image_tmp.shape[1]]))
            zstart = int(max([center[2] * scales[i][2] - patch_size[2] / 2, 0]))
            zend = int(min([center[2] * scales[i][2] + patch_size[2] / 2, image_tmp.shape[2]]))

            image_tmp = image_tmp[ystart:yend, xstart:xend, zstart:zend]
            if i == 0:
                label_map_tmp = label_map_tmp[ystart:yend, xstart:xend, zstart:zend]

            ystart = int((patch_size[0] - image_tmp.shape[0]) / 2)
            yend = int(ystart + image_tmp.shape[0])
            xstart = int((patch_size[1] - image_tmp.shape[1]) / 2)
            xend = int(xstart + image_tmp.shape[1])
            zstart = int((patch_size[2] - image_tmp.shape[2]) / 2)
            zend = int(zstart + image_tmp.shape[2])

            image_final[ystart:yend, xstart:xend, zstart:zend, i] = image_tmp
            if i == 0:
                label_map_final[ystart:yend, xstart:xend, zstart:zend] = label_map_tmp

        x.append(image_final)
        y.append(label_map_final)

    @staticmethod
    def post_process(gt1, pred1, gt2, pred2, gt3, pred3):
        m = 1
        t = 100
        t_tp = 0
        t_f = 0
        t_tp2 = 0
        t_f2 = 0
        dice = []

        c = gt1.shape[0]
        x = gt1.shape[1]
        y = gt1.shape[2]
        z = gt1.shape[3]
        print('==================================================================\n')
        for i in range(0, c):
            margin_pred1 = np.around(pred1[i, :, :, :, 0])
            margin_pred1[m:x - m, m:y - m, m:z - m] = np.zeros((x - 2 * m, y - 2 * m, z - 2 * m))

            margin_gt1 = np.around(gt1[i, :, :, :, 0])
            margin_gt1[m:x - m, m:y - m, m:z - m] = np.zeros((x - 2 * m, y - 2 * m, z - 2 * m))

            margin_pred2 = np.around(pred2[i, :, :, :, 0])
            margin_pred2[m:x - m, m:y - m, m:z - m] = np.zeros((x - 2 * m, y - 2 * m, z - 2 * m))

            margin_gt2 = np.around(gt2[i, :, :, :, 0])
            margin_gt2[m:x - m, m:y - m, m:z - m] = np.zeros((x - 2 * m, y - 2 * m, z - 2 * m))

            print('sample' + str(i) + ': ' + str(np.count_nonzero(margin_pred1)) + '--' + str(np.count_nonzero(margin_gt1))+'\n')

            if np.count_nonzero(margin_pred1) >= t or np.count_nonzero(margin_gt1) >= t:  # TRUE
                if np.count_nonzero(margin_pred2) >= t or np.count_nonzero(margin_gt2) >= t:
                    f = np.count_nonzero(np.add(gt3[i, :, :, :, 0], np.around(pred3[i, :, :, :, 0])) == 1)  # XOR
                    tp = np.count_nonzero(np.multiply(gt3[i, :, :, :, 0], np.around(pred3[i, :, :, :, 0])))  # AND
                    t_f += f*64
                    t_tp += tp*64

                    t_f2 += f
                    t_tp2 += tp
                    dice.append((2 * tp) / (f + 2 * tp))

                else:
                    f = np.count_nonzero(np.add(gt2[i, :, :, :, 0], np.around(pred2[i, :, :, :, 0])) == 1)  # XOR
                    tp = np.count_nonzero(np.multiply(gt2[i, :, :, :, 0], np.around(pred2[i, :, :, :, 0])))  # AND
                    t_f += f*8
                    t_tp += tp*8

                    t_f2 += f
                    t_tp2 += tp
                    dice.append((2 * tp) / (f + 2 * tp))
            else:
                f = np.count_nonzero(np.add(gt1[i, :, :, :, 0], np.around(pred1[i, :, :, :, 0])) == 1)  # XOR
                tp = np.count_nonzero(np.multiply(gt1[i, :, :, :, 0], np.around(pred1[i, :, :, :, 0])))  # AND
                t_f += f
                t_tp += tp

                t_f2 += f
                t_tp2 += tp
                dice.append((2 * tp) / (f + 2 * tp))

        return np.average(dice), (2 * t_tp) / (t_f + 2 * t_tp), (2 * t_tp2) / (t_f2 + 2 * t_tp2)
