from __future__ import print_function

import numpy as np
import math
import scipy.io
import os
from sklearn.model_selection import KFold
from xlrd import open_workbook


class DS:
    def __init__(self, path, patch_size, channel, K=5, angles=[], aug_scales=[]):
        print('DS - initialization')
        self.path = path
        self.patch_size = patch_size
        self.K = K
        self.angles = angles
        self.aug_scales = aug_scales
        self.channel = channel
        self.step_scales = []

        self.kf = KFold(n_splits=K, shuffle=True)
        self.train_indexes = []
        self.test_indexes = []

        self.label_maps = []
        self.images = []
        self.centers = []
        self.count = 0
        self.slice_counter = []

        self.load()
        self.generate_k_fold_indexes()

        # self.calculate_complexity()

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
        rejected = 0
        for j in scale:
            # print('DS - augmenting_zoom by scale: ', str(j))
            for i in range(0, count):
                # print('DS - zoom - image : ', i)
                new_image = scipy.ndimage.interpolation.zoom(images[i], j)
                new_map = np.around(scipy.ndimage.interpolation.zoom(label_maps[i], j))
                if np.count_nonzero(new_map) > 50:
                    images.append(new_image)
                    label_maps.append(new_map)
                    centers.append([round(centers[i][0] * j), round(centers[i][1] * j), round(centers[i][2] * j)])
                else:
                    rejected += 1
        return (count * (len(scale) + 1)) - rejected

    def augment_rotation(self, count, images, label_maps, centers):
        print('DS - augmenting__rotation')
        for i in range(0, count):
            # print('DS - rotation - image : ', i)
            for angle in self.angles:
                images.append(scipy.ndimage.rotate(images[i], angle=-angle))
                label_maps.append(scipy.ndimage.rotate(label_maps[i], angle=-angle))
                x, y = self.rotate(centers[i][0], centers[i][1], images[i].shape[0] / 2, images[i].shape[1] / 2,
                                   (-angle / 180) * math.pi)
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
        train_count = self.augment_zoom(train_count, train_image, train_label_map, train_center, self.aug_scales)
        # ==================================flip===========================================
        train_count = self.augment_flip(train_count, train_image, train_label_map, train_center)
        # ======================================================================================================
        y_train_2D = []
        x_train_2D = []

        tmp_x = []
        tmp_y = []

        for i in range(0, train_count):
            self.add(train_image[i], train_label_map[i], train_center[i], tmp_x, tmp_y)
        for i in range(0, len(tmp_x)):
            for j in range(tmp_x[i].shape[2]):
                if np.count_nonzero(tmp_y[i][:, :, j]) > 0:
                    x_train_2D.append(tmp_x[i][:, :, j, :])
                    y_train_2D.append(tmp_y[i][:, :, j])

        x_train = np.reshape(np.array(x_train_2D),
                             (len(x_train_2D), patch_size[0], patch_size[1], self.channel))
        y_train = np.reshape(np.array(y_train_2D), (len(y_train_2D), patch_size[0], patch_size[1], 1))
        # ===============t================e====================s=================t================================
        t_x = []
        t_y = []

        x_test_2D = []

        y_test_2D = []

        for i in self.test_indexes[fold]:
            self.add(self.images[i], self.label_maps[i], self.centers[i], t_x, t_y)
        for i in range(0, len(t_x)):
            hamed = 0
            for j in range(t_x[i].shape[2]):
                if (np.count_nonzero(t_y[i][:, :, j]) > 0):
                    x_test_2D.append(t_x[i][:, :, j, :])
                    y_test_2D.append(t_y[i][:, :, j])
                    hamed += 1
            self.slice_counter.append(hamed)
            print(str(hamed))

        x_test_2D = np.reshape(np.array(x_test_2D), (len(x_test_2D), patch_size[0], patch_size[1], self.channel))
        y_test_2D = np.reshape(np.array(y_test_2D), (len(y_test_2D), patch_size[0], patch_size[1], 1))

        x_test.append(x_test_2D)
        y_test.append(y_test_2D)

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
        image_final = np.ones((int(patch_size[0]), int(patch_size[1]), image.shape[2], self.channel))
        label_map_final = np.zeros((int(patch_size[0]), int(patch_size[1]), image.shape[2]))

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
            zstart = int(max([center[2] * scales[i][2] - image.shape[2] / 2, 0]))
            zend = int(min([center[2] * scales[i][2] + image.shape[2] / 2, image_tmp.shape[2]]))

            image_tmp = image_tmp[ystart:yend, xstart:xend, zstart:zend]
            if i == 0:
                label_map_tmp = label_map_tmp[ystart:yend, xstart:xend, zstart:zend]

            ystart = int((patch_size[0] - image_tmp.shape[0]) / 2)
            yend = int(ystart + image_tmp.shape[0])
            xstart = int((patch_size[1] - image_tmp.shape[1]) / 2)
            xend = int(xstart + image_tmp.shape[1])
            zstart = int((image.shape[2] - image_tmp.shape[2]) / 2)
            zend = int(zstart + image_tmp.shape[2])

            image_final[ystart:yend, xstart:xend, zstart:zend, i] = image_tmp
            if i == 0:
                label_map_final[ystart:yend, xstart:xend, zstart:zend] = label_map_tmp

        x.append(image_final)
        y.append(label_map_final)

    def post_process2(self, fold, logger, x_test, y_test, model):
        m = 1  # margin
        c = y_test[0].shape[0]
        pred = model.predict(x_test[0])
        gt = y_test[0]
        acc_tp = 0
        acc_fp = 0
        acc_fn = 0

        sample_index =0
        slice_index = 0
        for i in range(0, c):
            print(str(i), '--', str(slice_index), ' of ', str(self.slice_counter[sample_index]))
            temp_pred = np.around(pred[i, :, :, 0])
            temp_gt = gt[i, :, :, 0]

            acc_tp += np.count_nonzero(np.multiply(temp_gt, temp_pred))  # AND
            acc_fp += np.count_nonzero(np.bitwise_and(temp_gt == 0, temp_pred == 1))
            acc_fn += np.count_nonzero(np.bitwise_and(temp_gt == 1, temp_pred == 0))

            if self.slice_counter[sample_index]-1 == slice_index:
                dice = (2 * acc_tp) / ((acc_fp + acc_fn) + 2 * acc_tp)
                logger.write(str(dice) + "\n")
                print(str(dice))
                acc_tp = 0
                acc_fp = 0
                acc_fn = 0
                sample_index += 1
                slice_index = -1




            slice_index += 1



