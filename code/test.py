import cv2 as cv

img = cv.imread('../img_graycard_6mm/3.png' )

# undistort
cv_file = cv.FileStorage("calib_6.xml", cv.FILE_STORAGE_READ)
cam_matrix = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist").mat()
print("cam_matrix\n", cam_matrix)
print("dist\n", dist)
cv_file.release()
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(cam_matrix, dist, (w,h), 1, (w,h))
image_undistorted = cv.undistort(img, cam_matrix, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = image_undistorted[y:y+h, x:x+w]

image = dst
cv.imshow("img undistorted", image)
cv.waitKey(0)
cv.imwrite("../img_cal_area_6mm/undistorted.png", image)
