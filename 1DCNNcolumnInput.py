#Score:0.47948

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# import mnist data and visualize first image
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.load("/kaggle/input/audio-binary-classification/train_data.npy")
X_test = np.load("/kaggle/input/audio-binary-classification/test_data.npy")
y_train = pd.read_csv('/kaggle/input/audio-binary-classification/train_labels.csv').loc[:,'Label'].values


import numpy as np
import pandas as pd
# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten, Dense, Dropout

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#n_timesteps, n_features = X_train.shape[0], X_train.shape[1]
n_timesteps = X_train.shape[1]
n_features = 1

# Initialising the CNN
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
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
df2.to_csv("y_pred_cnn_1d_shape1_1.csv", index=False)