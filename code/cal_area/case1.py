import cv2 as cv
import cv2
import numpy as np

def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img

image = cv.imread("../../img/2.png")
# undistort
cv_file = cv.FileStorage("../calib_6.xml", cv.FILE_STORAGE_READ)
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
cv2.imshow("img undistorted", image)
cv2.waitKey(0)

image = center_crop(image,(500,400))
cv2.imshow("img drop", image)
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

height, width = mask_blue.shape
print("h, w :", height, width)
nzcount_blue = cv2.countNonZero(mask_blue)

nzcount_red = cv2.countNonZero(mask_red)
print(nzcount_blue, nzcount_red)
x  = nzcount_red / (nzcount_blue)
print(x)
area_ref = 5 * 5
area_cal = 23.5 * 24.6
print(area_cal)
print("the area :", x*area_ref)
# cv.imwrite("../images/mask1_20.jpg", mask_blue)
# cv.imwrite("../images/mask2_20.jpg", mask_red)
# cv2.imshow('maskb', mask_blue)
# cv2.imshow('maskr', mask_red)
cv2.imshow('resultb', result_blue)
cv2.imshow('resultr', result_red)
# crop_img = mask_blue[0:0+300, 0:0+650]
# cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
# cv2.imshow('result', result)

