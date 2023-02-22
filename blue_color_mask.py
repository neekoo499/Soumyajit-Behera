# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:07:22 2023

@author: neeko
"""


import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #Red color
    low_blue = np.array([94, 80, 2]) # lowest hue would be - 161,155,84
    high_blue = np.array([126, 255, 255])
    
    blue_mask = cv2.inRange(hsv_frame,low_blue,high_blue)
    #inRange() returns a binary image which is white where the colors are detected  and zero otherwise
    blue = cv2.bitwise_and(frame, frame, mask = blue_mask)
    #bitwise_and returns an array that corresponds to the resulting image from the merge of two given image
    cv2.imshow('Frame',frame)
    cv2.imshow('Blue',blue)
    
    key = cv2.waitKey(1)
    if key == 'q':
        break
    