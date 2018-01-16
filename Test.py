import cv2
import numpy as np
img = cv2.imread('D:\\1.jpg',0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ))
cv2.imshow('image',res)
cv2.waitKey(0)