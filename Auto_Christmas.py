import cv2
import sys
import logging as log
import datetime as dt
cascPath = "./haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
# 打开视频捕获设备
video_capture = cv2.VideoCapture(0)##可改为视频
hat = cv2.imread('hat.png')#,cv2.IMREAD_UNCHANGED)
# cv2.imshow('hat',hat)
# hat = cv2.resize(hat,(300,300))
# cv2.imshow('hat1',hat)
# cv2.waitKey(0)
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
    # 读视频帧
    ret, frame = video_capture.read()
    # 转为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 调用分类器进行检测
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(30, 30),)

    # 画矩形框
    for (x, y, w, h) in faces:
        hat = cv2.resize(hat,(w,w))
        frame[y-w:y,x:x+w] = hat
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # 显示视频
    cv2.imshow('Video', frame)
    print(len(faces))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 关闭摄像头设备
video_capture.release()
# 关闭所有窗口
cv2.destroyAllWindows()

