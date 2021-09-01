# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:20:09 2019

@author: Samira
"""
#Leaderboard Score: 0.98138

# Convolutional Neural Network

import numpy as np
import pandas as pd
# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras import regularizers
# import mnist data and visualize first image
X_train = np.load("/kaggle/input/audio-binary-classification/train_data.npy")
X_test = np.load("/kaggle/input/audio-binary-classification/test_data.npy")
y_train = pd.read_csv('/kaggle/input/audio-binary-classification/train_labels.csv').loc[:,'Label'].values
"""
X_train = np.load("audio-binary-classification/train_data.npy")
X_test = np.load("audio-binary-classification/test_data.npy")
y_train = pd.read_csv('audio-binary-classification/train_labels.csv').loc[:,'Label'].values

"""

#n_timesteps, n_features = X_train.shape[0], X_train.shape[1]
n_timesteps = 441
n_features = 100

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = np.reshape(X_train, (-1, 441, 100, 1))
X_test = np.reshape(X_test, (X_test.shape[0], n_timesteps, n_features, 1))

# Initialising the CNN
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(10,10), activation='relu', input_shape=(n_timesteps,n_features, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_regularizer = regularizers.l1(0.01)))
model.add(Dense(50, activation='relu', kernel_regularizer = regularizers.l1(0.01)))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 100, epochs = 50)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = model.predict(X_test)
#np.savetxt("y_pred_ann.csv", y_pred, delimiter=",")
  
df2 = pd.DataFrame(y_pred, index = range(y_pred.shape[0]))
df2.reset_index(level=0, inplace=True)
df2.columns=['Id', 'Label']
df2.to_csv("y_pred_128cnn_2D_100_features.csv", index=False)
