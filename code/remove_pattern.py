import cv2 as cv
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

originalImage  = cv.imread('../../img_cal_area_6mm/8.png')
cv.imshow("originalImage", originalImage)
cv.waitKey(0)
print(originalImage.shape)
originalImage = cv.cvtColor(originalImage , cv.COLOR_BGR2RGB)
crop_img = originalImage[100:(720-100), 280:(1280-280)]
print(crop_img.shape)
cv.imshow("Cropped", crop_img)

img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

plt.hist(img.ravel(),256,[0,256])
plt.show()

T, threshInv = cv2.threshold(img, 85, 255,
	cv2.THRESH_BINARY_INV)

# T, threshInv = cv2.threshold(img, 60, 255,
# 	cv2.THRESH_BINARY_INV)

cv2.imshow("Threshold Binary Inverse", threshInv)

#
# cv.imshow("Results", th1)

cv.waitKey(0)