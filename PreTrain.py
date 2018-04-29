import os
import scipy.io
import numpy as np


class PreTrain:
    @staticmethod
    def getdata(width, margin):

        data = []
        counter = 0

        fileList = os.listdir(".\matpdata")
        for file in fileList:
            print(".\matpdata\\" + file)
            image = scipy.io.loadmat(".\matpdata\\" + file)
            image = image['X']
            image = np.reshape(image, (image.shape[0], image.shape[1], image.shape[3]))

            for x in range(width[0] + margin, image.shape[0] - (width[0] + margin), width[0] * 2):
                for y in range(width[1] + margin, image.shape[1] - (width[1] + margin), width[1] * 2):
                    for z in range(width[2] + margin, image.shape[2] - (width[2] + margin), width[2] * 2):
                        data.append(
                            image[x - width[0]:x + width[0], y - width[1]:y + width[1], z - width[2]:z + width[2]])
                        counter += 1
                        if counter%100==0:
                            print(counter)

        return np.reshape(np.array(data), (counter, width[0]*2, width[1]*2, width[2]*2, 1))
