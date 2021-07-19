from PIL import Image
import cv2 as cv
import numpy as np
import os

# img = cv.imread('img_new.jpg')
# crop_img = img[555:555+200, 155:155+200]
# avgR = int(np.mean(crop_img[:, :, 2]))
# cv.imshow("cropped", crop_img)
# cv.waitKey(0)
# print(crop_img)
img = cv.imread('../img_graycard_6mm/0.png' )
crop_img = img[555:555+200, 155:155+200]
# cv.imshow("cropped", crop_img)
# cv.waitKey(0)
avgR = int(np.mean(crop_img[:, :, 2]))
avgG = int(np.mean(crop_img[:, :, 1]))
avgB = int(np.mean(crop_img[:, :, 0]))
print(avgB, avgG, avgR)
brightnessR = avgR - 119
brightnessG = avgG - 119
brightnessB = avgB - 119

# data = []  # using an array is more convenient for tabulate.
#
# img = cv.imread('../calibration_image/gray.png' )
#
# avgR = int(np.mean(img[:, :, 2]))
# avgG = int(np.mean(img[:, :, 1]))
# avgB = int(np.mean(img[:, :, 0]))
#
# print("avg :", avgR, avgG, avgB)
#
# img = Image.open('../calibration_image/gray.png')
# pixels = img.load()
# img_new = Image.new(img.mode, img.size)
# pixels_new = img_new.load()
# brightnessR =  avgR/119
# brightnessB =  avgB/119
# brightnessG =  avgG/119
# print(brightnessR, brightnessB, brightnessG)
# for i in range(img_new.size[0]):
#     for j in range(img_new.size[1]):
#         r, b, g = pixels[i, j]
#         _r = int(r / brightnessR)
#         _b = int(b / brightnessB)
#         _g = int(g / brightnessG)
#         pixels_new[i, j] = (_r, _b, _g, 255)
# img_new.show()


img = Image.open('../calibration_image/28.png')
pixels = img.load()
img_new = Image.new(img.mode, img.size)
pixels_new = img_new.load()
for i in range(img_new.size[0]):
    for j in range(img_new.size[1]):
        r, b, g = pixels[i, j]
        _r = int(r - brightnessR)
        _b = int(b - brightnessB)
        _g = int(g - brightnessG)
        pixels_new[i, j] = (_r, _b, _g, 255)
img_new.show()



