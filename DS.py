from __future__ import print_function

import numpy as np
import math
import scipy.io
import os
from sklearn.model_selection import KFold
from xlrd import open_workbook


class DS:
    def __init__(self, path, patch_size, channel, K=5, angles=[], aug_scales=[], pp_scales=[]):
        print('DS - initialization')
        self.path = path
        self.patch_size = patch_size
        self.K = K
        self.angles = angles
        self.aug_scales = aug_scales
        self.pp_scales = pp_scales
        self.channel = channel

        self.kf = KFold(n_splits=K, shuffle=True)
        self.train_indexes = []
        self.test_indexes = []

        self.label_maps = []
        self.images = []
        self.centers = []
        self.count = 0

        self.load()
        self.generate_k_fold_indexes()

        # self.calculate_pred_complexity()

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
        train_count = self.augment_zoom(train_count, train_image, train_label_map, train_center, self.aug_scales)
        # ================================rotation=========================================
        train_count = self.augment_rotation(train_count, train_image, train_label_map, train_center)
        # ==================================flip===========================================
        train_count = self.augment_flip(train_count, train_image, train_label_map, train_center)
        # ======================================================================================================
        for i in range(0, train_count):
            self.add(train_image[i], train_label_map[i], train_center[i], x_train, y_train)
        x_train = np.reshape(np.array(x_train),
                             (len(x_train), patch_size[0], patch_size[1], patch_size[2], self.channel))
        y_train = np.reshape(np.array(y_train), (len(x_train), patch_size[0], patch_size[1], patch_size[2], 1))
        # ===============t================e====================s=================t================================
        t_x = []
        t_y = []
        for i in self.test_indexes[fold]:
            self.add(self.images[i], self.label_maps[i], self.centers[i], t_x, t_y)

        t_x = np.reshape(np.array(t_x), (len(t_x), patch_size[0], patch_size[1], patch_size[2], self.channel))
        t_y = np.reshape(np.array(t_y), (len(t_y), patch_size[0], patch_size[1], patch_size[2], 1))
        x_test.append(t_x)
        y_test.append(t_y)

        for j in self.pp_scales:
            t_x = []
            t_y = []
            for i in self.test_indexes[fold]:
                self.add(scipy.ndimage.interpolation.zoom(self.images[i], j),
                         np.around(scipy.ndimage.interpolation.zoom(self.label_maps[i], j)),
                         [round(self.centers[i][0] * j), round(self.centers[i][1] * j),
                          round(self.centers[i][2] * j)], t_x, t_y)
            t_x = np.reshape(np.array(t_x), (len(t_x), patch_size[0], patch_size[1], patch_size[2], self.channel))
            t_y = np.reshape(np.array(t_y), (len(t_y), patch_size[0], patch_size[1], patch_size[2], 1))
            x_test.append(t_x)
            y_test.append(t_y)

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

    def post_process2(self, fold, logger, x_test, y_test, model):
        m = 1  # margin
        c = y_test[0].shape[0]

        for i in range(0, c):
            scale_index = 0
            while True:
                pred = model.predict(x_test[scale_index])
                gt = y_test[scale_index]
                x = gt.shape[1]
                y = gt.shape[2]
                z = gt.shape[3]

                temp_pred = DS.round(pred[i, :, :, :, 0], 0.5)
                margin_pred = DS.round(pred[i, :, :, :, 0], 0.5)
                margin_pred[m:x - m, m:y - m, m:z - m] = np.zeros((x - 2 * m, y - 2 * m, z - 2 * m))

                temp_gt = gt[i, :, :, :, 0]
                margin_gt = np.around(gt[i, :, :, :, 0])
                margin_gt[m:x - m, m:y - m, m:z - m] = np.zeros((x - 2 * m, y - 2 * m, z - 2 * m))

                tp = np.count_nonzero(np.multiply(temp_gt, temp_pred))  # AND
                fp = np.count_nonzero(np.bitwise_and(temp_gt == 0, temp_pred == 1))
                fn = np.count_nonzero(np.bitwise_and(temp_gt == 1, temp_pred == 0))

                comp = self.complexity(temp_gt)
                comp_pred = self.complexity(temp_pred)

                dice = (2 * tp) / ((fp + fn) + 2 * tp)

                if np.count_nonzero(margin_pred) == 0 or len(self.pp_scales) == scale_index + 1:
                    logger.write(str(self.test_indexes[fold][i]) + "," + str(scale_index)+ str(
                        dice) + "," + str(comp) + ","+ str(comp_pred) +"\n")
                    break
                else:
                    scale_index += 1

    def complexity(self, gt):
        a = 0
        # gt = self.label_maps[counter]
        for i in range(1, gt.shape[0] - 1):
            for j in range(1, gt.shape[1] - 1):
                for k in range(1, gt.shape[2] - 1):
                    if gt[i, j, k] > 0 and (gt[i + 1, j, k] < 1 or gt[i - 1, j, k] < 1 or gt[i, j + 1, k] < 1 or gt[
                        i, j - 1, k] < 1 or gt[i, j, k + 1] < 1 or gt[i, j, k - 1] < 1):
                        a += 1

        v = np.count_nonzero(gt[:, :, :] > 0)

        return float(a ** 3) / float(v ** 2)

    def calculate_gt_complexity(self):
        print("DS - Calculating gt complexity")
        file = open('\comp.txt', "w+")
        for i in range(0, self.count):
            complexity = str(self.complexity(self.label_maps[i]))
            file.write(str(i) + ", " + complexity)
            print("complexity for: " + str(i) + ", " + complexity)

    @staticmethod
    def round(value, th):
        result = np.zeros_like(value)
        for i in range(0, value.shape[0]):
            for j in range(0, value.shape[1]):
                for k in range(0, value.shape[2]):
                    if value[i, j, k] > th:
                        result[i, j, k] = 1
                    else:
                        result[i, j, k] = 0
        return result
