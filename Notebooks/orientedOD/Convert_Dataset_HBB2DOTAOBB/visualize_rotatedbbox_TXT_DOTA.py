import cv2
import math

img =cv2.imread(r'D:\SourceCode\AIO_2022\Pill_Recognition_Challenge\segmentation_preprocess\test3.png')

#ROTATED BOUNDING BOXES IN TXT
(x_left_top,y_left_top,x_right_top, y_right_top,x_right_bottom,y_right_bottom,x_left_bottom,y_left_bottom)=(42, 85, 66, 25, 97, 37,73, 98)

cv2.line(img,(x_left_top,y_left_top),(x_right_top,y_right_top),(255, 0, 0), 1, 1)
cv2.line(img,(x_left_top,y_left_top),(x_left_bottom,y_left_bottom),(255, 0, 0), 1, 1)
cv2.line(img,(x_left_bottom,y_left_bottom),(x_right_bottom,y_right_bottom),(255, 0, 0), 1, 1)
cv2.line(img,(x_right_bottom,y_right_bottom),(x_right_top,y_right_top),(255, 0, 0), 1, 1)
cv2.imshow('ok',img)

cv2.waitKey(0)
