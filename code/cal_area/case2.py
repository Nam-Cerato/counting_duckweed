import cv2 as cv
import cv2
import numpy as np
import argparse

originalImage  = cv.imread('image/calibresult.png')
# cv.imshow("originalImage", originalImage)
# cv.waitKey(0)
originalImage = cv.cvtColor(originalImage , cv.COLOR_BGR2RGB)
# cv.imshow("originalImage", originalImage)
# cv.waitKey(0)

reshapedImage = np.float32(originalImage.reshape(-1, 3))

numberOfClusters = 4

stopCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.1)

ret, labels, clusters = cv.kmeans(reshapedImage, numberOfClusters, None, stopCriteria, 10, cv.KMEANS_RANDOM_CENTERS)

clusters = np.uint8(clusters)

intermediateImage = clusters[labels.flatten()]
clusteredImage = intermediateImage.reshape((originalImage.shape))
print(clusteredImage)

cv.imwrite("image/clusteredImage.jpg", clusteredImage)

removedCluster = 1

cannyImage = np.copy(originalImage).reshape((-1, 3))
cannyImage[labels.flatten() == removedCluster] = [0, 0, 0]

cannyImage = cv.Canny(cannyImage,100,200).reshape(originalImage.shape)
cv.imwrite("image/cannyImage.jpg", cannyImage)

initialContoursImage = np.copy(cannyImage)
imgray = cv.cvtColor(initialContoursImage, cv.COLOR_BGR2GRAY)
cv.imwrite("image/imgray.jpg", cannyImage)
_, thresh = cv.threshold(imgray, 50, 255, 0)
cv.imwrite("image/thresh.jpg", thresh)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(initialContoursImage, contours, -1, (0,0,255), cv.CHAIN_APPROX_SIMPLE)
cv.imwrite("image/initialContoursImage.jpg", initialContoursImage)

cnt = contours[0]
largest_area=0
index = 0
for contour in contours:
    if index > 0:
        area = cv.contourArea(contour)
        if (area>largest_area):
            largest_area=area
            cnt = contours[index]
    index = index + 1

biggestContourImage = np.copy(originalImage)
cv.drawContours(biggestContourImage, [cnt], -1, (0,0,255), 3)
cv.imwrite("../images/biggestContourImage.jpg", biggestContourImage)





# calculate pixel object

# src = cv.imread("../images/calibresult.png", cv.IMREAD_GRAYSCALE)
# ret, image_edit = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#
# cv.imwrite("../images/binary.jpg", image_edit)
# cv.imshow("binary", image_edit)
# cv.waitKey(0)
# height, width  = image_edit.shape
# print(height, width)
# nzcount = cv2.countNonZero(image_edit)
# print(nzcount)
# print("percentage :", nzcount)




