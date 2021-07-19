import cv2 as cv
import cv2
import numpy as np


image = cv.imread("../../img_cal_area_6mm/0.png")
# undistort
cv_file = cv.FileStorage("../test.xml", cv.FILE_STORAGE_READ)
cam_matrix = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist").mat()
print("cam_matrix\n", cam_matrix)
print("dist\n", dist)
cv_file.release()
h,  w = image.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(cam_matrix, dist, (w,h), 1, (w,h))
image_undistorted = cv.undistort(image, cam_matrix, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = image_undistorted[y:y+h, x:x+w]
# cv.imwrite('../images/5.png', dst)
image = dst
cv2.imshow("img", image)
cv2.waitKey(0)

result = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_blue = np.array([94,80,2])
upper_blue = np.array([120,255,255])
# lower_blue = np.array([94,80,2])
# upper_blue = np.array([126,255,255])
lower_red = np.array([155,25,0])
upper_red = np.array([179,255,255])

mask_blue = cv2.inRange(image, lower_blue, upper_blue)
result_blue = cv2.bitwise_and(result, result, mask=mask_blue)

mask_red = cv2.inRange(image, lower_red, upper_red)
result_red = cv2.bitwise_and(result, result, mask=mask_red)
# cv.imwrite("../images/mask1_20.jpg", mask_blue)
# cv.imwrite("../images/mask2_20.jpg", mask_red)
cv2.imshow('maskb', mask_blue)
cv2.imshow('maskr', mask_red)
crop_img = mask_blue[0:0+300, 0:0+650]
# cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
# cv2.imshow('result', result)

height, width = mask_blue.shape
print("h, w :", height, width)
nzcount_blue = cv2.countNonZero(mask_blue)

nzcount_red = cv2.countNonZero(mask_red)
print(nzcount_blue, nzcount_red)
x  = nzcount_red / (nzcount_blue)
print( x)
area_ref = 5 * 5
area_cal = 23.5 * 24.6
print(area_cal)
print("the area :",x * area_ref)