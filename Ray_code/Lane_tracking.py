import cv2 as cv
import numpy as np

class lane_detecting:
    def __init__(self, img):
        self.img = img

    def lane_gray_edge(self):
        lane_gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        lane_blur = cv.GaussianBlur(lane_gray, (5, 5), 0)
        lane_edge = cv.Canny(lane_blur, 100, 200)
        return lane_edge
    
    def lane_mask_edge(self, h_min, h_max, s_min, s_max, v_min, v_max):
        lane_hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = cv.inRange(lane_hsv, lower, upper)
        result = cv.bitwise_and(self.img, self.img, mask = mask)
        return result
    
lane = cv.imread('Lane.jpg')
lane = cv.GaussianBlur(lane, (5, 5), 0)
image = lane_detecting(lane)
res = image.lane_mask_edge(0, 179, 0, 255, 0, 255)

cv.imshow('lane edge', image.lane_gray_edge())
cv.imshow('lane mask', res)
cv.waitKey(0)


