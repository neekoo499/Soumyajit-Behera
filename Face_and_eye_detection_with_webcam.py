# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:56:20 2023

@author: neeko
"""
import cv2

face_cascade = cv2.CascadeClassifier(r"D:\DATA_SCIENCE_CLASS_DOCUMENTS\Computer-Vision-Tutorial-master_8th_sept\Haarcascades\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"D:\DATA_SCIENCE_CLASS_DOCUMENTS\Computer-Vision-Tutorial-master_8th_sept\Haarcascades\haarcascade_eye.xml")

def detect(gray,frame):
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return frame

#Here we are doing face recognition by using webcam
video_capture = cv2.VideoCapture(0)
#VideoCapture - here we accessing your camera.when parameter is '0' then we are accessing internal camera of the computer
#when parameter is '1' we are accessing external camera of the computer
while True:
    _,frame = video_capture.read()
    #read() returns two object i.e _ and frame we only concentrate on frame
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
