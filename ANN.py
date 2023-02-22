# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:20:49 2023

@author: neeko
"""


import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

dataset = pd.read_csv(r"D:\DATA_SCIENCE_CLASS_DOCUMENTS\1st_spt\1st\ANN_ 1st\Churn_Modelling.csv")
x = dataset.iloc[:,3:].values
y = dataset.iloc[:,-1].values

#Encoding the catagorical data
#Label Encoding to the 'Gender' column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[1])],remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

#feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#splitting the dataset to train and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 0)

#Part 2 - Building the ANN
ann = tf.keras.models.Sequential()
#Adding input layer and first input layer
ann.add(tf.keras.layers.Dense(units = 6,activation = 'relu'))
#Adding another hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

#Adding the output layer
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


#Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])
#Training the ann on training set
ann.fit(x_train,y_train,batch_size = 32, epochs = 100)


#Predicting the test results
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Making the confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)

ac = accuracy_score(y_test,y_pred)
print(ac)
