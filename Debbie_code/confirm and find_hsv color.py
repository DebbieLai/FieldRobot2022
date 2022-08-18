from re import S
import cv2
import numpy as np

# 設定全域變數(這裡可以再修改)
step = 1
color = 0

# 這裡需要注意因影像捕捉非常即時，在切換畫面時非目標圖形會不小心變成目標圖形，所以要看前兩次是否都是目標的圖形。
# 或是可以改成取近10次的圖形
a1 = 0
a2 = 0
a3 = 0
i = 0
correct = 0


def empty(v):
    pass


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

        Area = cv2.contourArea(cnt)

        # 過濾雜訊
        if Area > 1000:
            # 找頂點
            peri = cv2.arcLength(cnt, True)
            vertices = cv2.approxPolyDP(cnt, peri*0.04, True)
            corners = len(vertices)

            # 框出圖形四角
            x, y, w, h = cv2.boundingRect(vertices)

            # 找出中點
            i = int((2*x+w)/2)  # 注意要把他們轉換成int才能丟入函式
            j = int((2*y+h)/2)
            # 找到HSV值
            hue, sat, val = hsv[j, i]  # 超級重要，先y方向再x方向
            # print(corners)

            # 查看是否為正確的corners
            if i % 3 == 0:
                a1 = corners
            if i % 3 == 1:
                a2 = corners
            if i % 3 == 2:
                a3 = corners
            i += 1
            if a1 == a2 and a3 == a2 and a1 == a3 and a1 == 4:
                correct = 1

            # 辨別指示顏色
            if step == 1 and correct == 1:
                if hue < 15:
                    if step == 1:
                        print('finish!')
                        step = 2
                        color = 1
                if hue > 20 and hue < 40:
                    if step == 1:
                        print('finish!')
                        step = 2
                        color = 2
                if hue > 55 and hue < 80:
                    if step == 1:
                        print('finish!')
                        step = 2
                        color = 3
                if hue > 85 and hue < 100:
                    if step == 1:
                        print('finish!')
                        step = 2
                        color = 4

            # 辨別目標顏色並圈起圖案
            if step == 2:
                if color == 1 and hue < 15:
                    cv2.drawContours(frame, cnt, -1, (255, 0, 0), 3)
                    cv2.circle(frame, (i, j), 5, (255, 0, 0), cv2.FILLED)
                if color == 2 and hue > 20 and hue < 40:
                    cv2.drawContours(frame, cnt, -1, (255, 0, 0), 3)
                    cv2.circle(frame, (i, j), 5, (255, 0, 0), cv2.FILLED)
                if color == 3 and hue > 55 and hue < 80:
                    cv2.drawContours(frame, cnt, -1, (255, 0, 0), 3)
                    cv2.circle(frame, (i, j), 5, (255, 0, 0), cv2.FILLED)
                if color == 4 and hue > 85 and hue < 100:
                    cv2.drawContours(frame, cnt, -1, (255, 0, 0), 3)
                    cv2.circle(frame, (i, j), 5, (255, 0, 0), cv2.FILLED)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# 關閉相機
cam.release()
cv2.destroyAllWindows()
