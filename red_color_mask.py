# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:32:11 2023

@author: neeko
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #Red color
    low_red = np.array([161, 155, 84]) # lowest hue would be - 161,155,84
    high_red = np.array([179, 255, 255])
    
    red_mask = cv2.inRange(hsv_frame,low_red,high_red)
    #inRange() returns a binary image which is white where the colors are detected  and zero otherwise
    red = cv2.bitwise_and(frame, frame, mask = red_mask)
    #bitwise_and returns an array that corresponds to the resulting image from the merge of two given image
    cv2.imshow('Frame',frame)
    cv2.imshow('Red',red)
    
    key = cv2.waitKey(1)
    if key == 'q':
        break
    
    
    