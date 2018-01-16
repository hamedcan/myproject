import cv2
import numpy as np
img = cv2.imread('D:\\1.jpg',0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ))
print(equ.shape)
print(equ.__class__)
print(equ.dtype)
print(img)
print(equ)
cv2.imshow('image',res)
cv2.waitKey(0)