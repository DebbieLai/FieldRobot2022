from re import S
import cv2
import numpy as np


def empty(v):
    pass


# 設定全域變數(這裡可以再修改)
image = cv2.imread('black.png')
b, g, r = image[0, 0]


def get_RGBColorCode(img, x=0, y=0):
    b, g, r = img[y, x]
    return b, g, r


# 開啟鏡頭
cam = cv2.VideoCapture(0)
if not cam:
    print('cannot open camera')
    exit()

# 建立Trackbar
# cv2.namedWindow('Trackbar')
# cv2.resizeWindow('Trackbar', 640, 320)

# cv2.createTrackbar('Hue min', 'Trackbar', 0, 179, empty)
# cv2.createTrackbar('Hue Max', 'Trackbar', 179, 179, empty)
# cv2.createTrackbar('Sat min', 'Trackbar', 0, 255, empty)
# cv2.createTrackbar('Sat Max', 'Trackbar', 255, 255, empty)
# cv2.createTrackbar('Val min', 'Trackbar', 0, 255, empty)
# cv2.createTrackbar('Val Max', 'Trackbar', 255, 255, empty)

while True:
    ret, frame = cam.read()
    if not ret:
        print('cannot receive frame')
        break

    # 轉成hsv
    img = frame.copy()
    imgcontours = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Trackbar抓取顏色
    # H_min = cv2.getTrackbarPos('Hue min', 'Trackbar')
    # H_Max = cv2.getTrackbarPos('Hue Max', 'Trackbar')
    # S_min = cv2.getTrackbarPos('Sat min', 'Trackbar')
    # S_Max = cv2.getTrackbarPos('Sat Max', 'Trackbar')
    # V_min = cv2.getTrackbarPos('Val min', 'Trackbar')
    # V_Max = cv2.getTrackbarPos('Val Max', 'Trackbar')

    # 遮罩。得出圖形
    # lower = np.array([H_min, S_min, V_min])
    # upper = np.array([H_Max, S_Max, V_Max])
    lower = np.array([0, 103, 134])
    upper = np.array([99, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    # 找出邊緣
    canny = cv2.Canny(result, 100, 200)
    # 找出輪廓
    contours, hierarchy = cv2.findContours(
        canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        # cv2.drawContours(imgcontours, cnt, -1, (255, 0, 0), 3)
        Area = cv2.contourArea(cnt)

        # 過濾雜訊
        if Area > 1000:
            # print(Area)
            # 算字體大小與間距
            d1 = int(Area/150)
            d2 = int(Area/800)
            s = 2
            if Area < 8000:
                s = 2
            if Area > 8000:
                s = 3

            # 找頂點
            peri = cv2.arcLength(cnt, True)
            vertices = cv2.approxPolyDP(cnt, peri*0.03, True)
            corners = len(vertices)
            # 框出圖形四角
            x, y, w, h = cv2.boundingRect(vertices)

            # 判斷顏色
            i = int((2*x+w)/2)  # 注意要把他們轉換成int才能丟入函式
            j = int((2*y+h)/2)
            b, g, r = get_RGBColorCode(result, i, j)

            c_b = 0
            c_g = 0
            c_r = 0
            if b < 100 and g < 150 and r > 180:
                c_b = 30
                c_g = 81
                c_r = 237
                cv2.putText(frame, 'Orange', (x, y-d1),
                            cv2.FONT_HERSHEY_PLAIN, s, (c_b, c_g, c_r), s)
            if b < 100 and g > 100 and r > 100:
                c_b = 0
                c_g = 223
                c_r = 227
                cv2.putText(frame, 'Yellow', (x, y-d1),
                            cv2.FONT_HERSHEY_PLAIN, s, (c_b, c_g, c_r), s)
            if b < 100 and g > 100 and r < 80:
                c_b = 0
                c_g = 193
                c_r = 0
                cv2.putText(frame, 'Green', (x, y-d1),
                            cv2.FONT_HERSHEY_PLAIN, s, (c_b, c_g, c_r), s)
            if b > 100 and g > 100 and r < 100:
                c_b = 211
                c_g = 188
                c_r = 0
                cv2.putText(frame, 'Blue', (x, y-d1),
                            cv2.FONT_HERSHEY_PLAIN, s, (c_b, c_g, c_r), s)
            # 圈起圖案
            cv2.rectangle(frame, (x, y), (x+w, y+h), (c_b, c_g, c_r), 3)

            # 判斷形狀形狀，輸出結果
            if corners == 3:
                cv2.putText(frame, 'Triangle', (x, y-d2),
                            cv2.FONT_HERSHEY_PLAIN, s, (c_b, c_g, c_r), s)
            if corners == 4:
                cv2.putText(frame, 'Rectangle', (x, y-d2),
                            cv2.FONT_HERSHEY_PLAIN, s, (c_b, c_g, c_r), s)
            if corners == 5:
                cv2.putText(frame, 'Pentagon', (x, y-d2),
                            cv2.FONT_HERSHEY_PLAIN, s, (c_b, c_g, c_r), s)
            if corners > 6:
                cv2.putText(frame, 'Circle', (x, y-d2),
                            cv2.FONT_HERSHEY_PLAIN, s, (c_b, c_g, c_r), s)

    # cv2.imshow('shape', mask)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# 關閉相機
cam.release()
cv2.destroyAllWindows()
