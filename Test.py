import numpy as np
import scipy.io
from xlrd import open_workbook
path = '.\data\\'
centers = []
count = 0
data = []
size = 1
print('Weight Initialize - loading')
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

        label_map = scipy.io.loadmat('.\data\gtruth_' + volname + '_perim.mat')
        label_map = label_map['gtruth_perim']
        label_map = np.reshape(label_map, (label_map.shape[0], label_map.shape[1], label_map.shape[2]))
        nz = np.nonzero(label_map)

        x = np.max(nz[0]) - np.min(nz[0])
        y = np.max(nz[1]) - np.min(nz[1])
        z = np.max(nz[2]) - np.min(nz[2])
        print(str(x) + "," + str(y) + "," + str(z))

