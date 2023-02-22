# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:10:32 2023

@author: neeko
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #Red color
    low_green = np.array([25, 52, 72]) # lowest hue would be - 161,155,84
    high_green = np.array([102, 255, 255])
    
    green_mask = cv2.inRange(hsv_frame,low_green,high_green)
    #inRange() returns a binary image which is white where the colors are detected  and zero otherwise
    green = cv2.bitwise_and(frame, frame, mask = green_mask)
    #bitwise_and returns an array that corresponds to the resulting image from the merge of two given image
    cv2.imshow('Frame',frame)
    cv2.imshow('Green',green)
    
    key = cv2.waitKey(1)
    if key == 'q':
        break