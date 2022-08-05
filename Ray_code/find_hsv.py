import cv2 as cv
import numpy as np

lane = cv.imread('Lane.jpg')
lane = cv.resize(lane, (0, 0), fx = 0.25, fy = 0.25)
hsv = cv.cvtColor(lane, cv.COLOR_BGR2HSV)

def empty(v):
    pass
##設定調整視窗
cv.namedWindow('Find the value of hsv')
##設定視窗寬度
cv.resizeWindow('Find the value of hsv', 640, 320)    

cv.createTrackbar('Hue Min', 'Find the value of hsv', 0, 179, empty)
cv.createTrackbar('Hue Max', 'Find the value of hsv', 179, 179, empty)
cv.createTrackbar('Sat Min', 'Find the value of hsv', 0, 255, empty)
cv.createTrackbar('Sat Max', 'Find the value of hsv', 255, 255, empty)
cv.createTrackbar('Val Min', 'Find the value of hsv', 0, 255, empty)
cv.createTrackbar('Val Max', 'Find the value of hsv',255, 255, empty)

while True:
    h_min = cv.getTrackbarPos('Hue Min', 'Find the value of hsv')
    h_max = cv.getTrackbarPos('Hue Max', 'Find the value of hsv')
    s_min = cv.getTrackbarPos('Sat Min', 'Find the value of hsv')
    s_max = cv.getTrackbarPos('Sat Max', 'Find the value of hsv')
    v_min = cv.getTrackbarPos('Val Min', 'Find the value of hsv')
    v_max = cv.getTrackbarPos('Val Max', 'Find the value of hsv')
    print(h_min, h_max, s_min, s_max, v_min, v_max)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv.inRange(hsv, lower, upper)
    cv.imshow('lane' , lane)
    cv.imshow('hsv' , hsv)
    cv.imshow('mask', mask)
    cv.waitKey(1)