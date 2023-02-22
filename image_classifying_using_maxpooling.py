# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:35:09 2023

@author: neeko
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

img = image.load_img(r"D:\DL_rev\training\happy\3.jpg")
plt.imshow(img)

i1 = cv2.imread(r"D:\DL_rev\training\happy\3.jpg")
i1
# 3 dimension metrics are created for the image
# the value ranges from 0-255

i1.shape
# shape of your image height, weight, rgb

train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)
# to scale all the images i need to divide with 255
# we need to resize the image using 200, 200 pixel

train_dataset = train.flow_from_directory(r"D:\DL_rev\training",
                                          target_size = (200,200),
                                          batch_size = 3,
                                          class_mode = 'binary')  #batch_size i.e in each iteration we will get 3 images

validation_dataset = validation.flow_from_directory(r"D:\DL_rev\validation",
                                          target_size = (200,200),
                                          batch_size = 3,
                                          class_mode = 'binary')

train_dataset.class_indices
train_dataset.classes

#Now we apply Max Pooling
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',input_shape = (200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),  # 3 filter we applied here
                                    
                                    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    
                                    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    
                                    tf.keras.layers.Flatten(),
                                    #Flatten() is the input does not affect the batch size
                                    
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    
                                    tf.keras.layers.Dense(1, activation = 'sigmoid'),
                                    
                                    ])

model.compile(loss = 'binary_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(lr = 0.001),
              metrics = ['accuracy'])

model_fit = model.fit(train_dataset,
                      steps_per_epoch = 3,
                      epochs = 10,
                      validation_data = validation_dataset)

dir_path = r"D:\DL_rev\testing"
for i in os.listdir(dir_path):
    print(i)
    
dir_path = r"D:\DL_rev\testing"
for i in os.listdir(dir_path):
    img = image.load_img(dir_path + '//' + i,target_size = (200,200))
    plt.imshow(img)
    plt.show()
    
    
dir_path = r"D:\DL_rev\testing"
for i in os.listdir(dir_path):
    img = image.load_img(dir_path + '//' + i,target_size = (200,200))
    plt.imshow(img)
    plt.show()
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    images = np.vstack([x])
#vstack() used to stack the sequence of input arrays vertically to make a single array.
    val = model.predict(images)
    if (val == 0):
        print('I am not Happy')
    else:
        print('I am Happy')