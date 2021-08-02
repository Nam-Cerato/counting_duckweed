import cv2 as cv
import cv2
import numpy as np
import argparse

def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img

originalImage  = cv.imread('../../img/2.png')
# cv.imshow("originalImage", originalImage)
# cv.waitKey(0)

originalImage =  center_crop(originalImage,(500,400))
cv2.imshow("img drop", originalImage)
cv2.waitKey(0)

# originalImage = cv.cvtColor(originalImage , cv.COLOR_BGR2RGB)
# cv.imshow("originalImage", originalImage)
# cv.waitKey(0)

cannyImage = cv.Canny(originalImage,10,200)
cv2.imshow("cannyImage", cannyImage)
cv2.waitKey(0)

contours, hierarchy = cv.findContours(originalImage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(originalImage, contours, -1, (0,0,255), cv.CHAIN_APPROX_SIMPLE)
# cv.imwrite("image/initialContoursImage.jpg", initialContoursImage)
cv2.imshow("initialContoursImage", originalImage)
cv2.waitKey(0)
with_contours = cv2.drawContours(originalImage, contours, -1, (255, 0, 255), 3)
print("contours : ", str(len(contours)))
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    print("pixel value: ",cv2.contourArea(c))
    # Make sure contour area is large enough
    if (cv2.contourArea(c)) > 10:
        cv2.rectangle(with_contours, (x, y), (x + w, y + h), (255, 0, 0), 5)

cv2.imshow("final", originalImage)
cv2.waitKey(0)



