import cv2

image = cv2.imread('test_image.jpg')
cv2.imshow('results', image)
cv2.waitKey(0)