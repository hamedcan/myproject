import random
import cv2
import numpy as np
import math
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from xlrd import open_workbook





def build_filters():
    filters = []
    ksize = 15
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    kern /= 1.5 * kern.sum()
    filters.append(kern)
    return filters


def gabour_filter(img):
    filters = build_filters()
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
    np.maximum(accum, fimg, accum)
    return accum


def gabour3D(image):
    for i in range(0, image.shape[2]):
        data = image[:, :, i]
        data *= 255
        data = data.astype(np.uint8)
        data = gabour_filter(data)
        data = data.astype(np.float64)
        data /= 255
        image[:, :, i] = data
    return image


def eqhist3D(image):
    for i in range(0, image.shape[2]):
        data = image[:, :, i]
        data *= 255
        data = data.astype(np.uint8)
        data = cv2.equalizeHist(data)
        data = data.astype(np.float64)
        data /= 255
        image[:, :, i] = data
    return image


def CLAHE3D(image):
    for i in range(0, image.shape[2]):
        data = image[:, :, i]
        data *= 255
        data = data.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        data = clahe.apply(data)
        data = data.astype(np.float64)
        data /= 255
        image[:, :, i] = data
    return image
















path = '.\data\\'
centers = []
count = 0
data = []
wb = open_workbook(path + 'data.xlsx')
for s in wb.sheets():
    nrows = s._dimnrows
    file_name = [''] * nrows
    for i in range(nrows):
        centers.append([int(s.cell(i, 4).value), int(s.cell(i, 3).value), int(s.cell(i, 5).value)])
        file_name[i] = s.cell(i, 0).value
    count = nrows

volname = file_name[25]

image = scipy.io.loadmat('.\data\enhanced_' + volname + '.mat')
image = image['enhanced']
image = np.reshape(image, (image.shape[0], image.shape[1], image.shape[3]))
# image = CLAHE3D(image)
# image = eqhist3D(image)
image = gabour3D(image)
plt.imshow(image[:,:,20], cmap='gray')
plt.show()
