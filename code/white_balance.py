from PIL import Image
import cv2 as cv
import numpy as np
import os

# img = cv.imread('../img_graycard_16mm/0.png' )
#
# # undistort
# cv_file = cv.FileStorage("calib_16.xml", cv.FILE_STORAGE_READ)
# cam_matrix = cv_file.getNode("camera_matrix").mat()
# dist = cv_file.getNode("dist").mat()
# print("cam_matrix\n", cam_matrix)
# print("dist\n", dist)
# cv_file.release()
# h,  w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(cam_matrix, dist, (w,h), 1, (w,h))
# image_undistorted = cv.undistort(img, cam_matrix, dist, None, newcameramtx)
# # crop the image
# x, y, w, h = roi
# dst = image_undistorted[y:y+h, x:x+w]
#
# image = dst
# cv.imshow("img undistorted", image)
# cv.waitKey(0)
# cv.imwrite("undistorted.png", image)

image = cv.imread("../img_cal_area_6mm/undistorted.png")
crop_img = image[20:20+150, 500:500+150]
# cv.imshow("cropped", crop_img)
# cv.waitKey(0)
avgR = int(np.mean(crop_img[:, :, 2]))
avgG = int(np.mean(crop_img[:, :, 1]))
avgB = int(np.mean(crop_img[:, :, 0]))
print(avgB, avgG, avgR)
brightnessR = float(avgR / 118)
brightnessG = float(avgG / 118)
brightnessB = float(avgB / 118)
print(brightnessB, brightnessG, brightnessR)

# lum = (brightnessG + brightnessR+ brightnessB) /3
height, width = image.shape[0], image.shape[1]

for i in range(height):
    for j in range(width):
        b, g, r = image[i, j]
        _r = int(r / brightnessR)
        _b = int(b / brightnessB)
        _g = int(g / brightnessG)
        # _r = int(int(r * lum) / brightnessR)
        # _b = int(int(b * lum) / brightnessB)
        # _g = int(int(g * lum) / brightnessG)
        image[i,j][0] = _b
        image[i,j][1] = _g
        image[i,j][2] = _r
cv.imshow("calib color", image)
cv.waitKey(0)

crop_img = image[20:20+150, 500:500+150]
cv.imshow("cropped", crop_img)
cv.waitKey(0)

avgR = int(np.mean(crop_img[:, :, 2]))
avgG = int(np.mean(crop_img[:, :, 1]))
avgB = int(np.mean(crop_img[:, :, 0]))
print(avgB, avgG, avgR)

cv.imwrite("../img_cal_area_6mm/white_balance.png", image)


# image = Image.open('../img_cal_area_6mm/undistorted.png')
# pixels = image.load()
# print(pixels)
# img_new = Image.new(image.mode, image.size)
# pixels_new = img_new.load()
# print(img_new.size[0], img_new.size[1], img_new.size)
# for i in range(img_new.size[0]):
#     for j in range(img_new.size[1]):
#         r, b, g = pixels[i, j]
#         _r = int(r / brightnessR)
#         _b = int(b / brightnessB)
#         _g = int(g / brightnessG)
#         pixels_new[i, j] = (_r, _b, _g, 255)
# img_new.show()


