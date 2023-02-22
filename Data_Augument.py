# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:12:35 2023

@author: neeko
"""

import tensorflow as tf

from keras_preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
datagen = ImageDataGenerator(rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

#import tensorflow as backend
img = load_img(r"C:\Users\neeko\Videos\istockphoto-479667835-612x612.jpg")

x = img_to_array(img) #its a numpy array of shape(407,612,3)
x = x.reshape((1,) + x.shape)
x
# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x,batch_size=1,save_to_dir = "D:\Deep_Learning\DATA ARGUMENT",save_prefix = 'elephant',save_format = 'jpeg'):
    i += 1
    if(i > 30):
        break