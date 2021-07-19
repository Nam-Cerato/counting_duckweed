import cv2 as cv
cv_file = cv.FileStorage("test.xml", cv.FILE_STORAGE_READ)
cam_matrix = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist").mat()
print("cam_matrix\n", cam_matrix)
print("dist\n", dist)
cv_file.release()