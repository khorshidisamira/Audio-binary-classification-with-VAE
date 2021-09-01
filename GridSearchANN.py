import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Part 1 - Data Preprocessing

# Importing the libraries
import matplotlib.pyplot as plt

# Importing the dataset
#dataset = pd.read_csv('Churn_Modelling.csv')
#X = dataset.iloc[:, 3:13].values
#y = dataset.iloc[:, 13].values

# import mnist data and visualize first image
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.load("/kaggle/input/audio-binary-classification/train_data.npy")
X_test = np.load("/kaggle/input/audio-binary-classification/test_data.npy")
y_train = pd.read_csv('/kaggle/input/audio-binary-classification/train_labels.csv').loc[:,'Label'].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer, neurons= 10, dropout_rate = 0.2):
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = neurons, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
    # classifier.add(Dropout(p = 0.1))
    
    # Adding the second hidden layer
    classifier.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.2))
    
    
    classifier.add(Dense(units = 250, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.2))
    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [100, 39],
              'epochs': [100, 50],
              'neurons':[100, 500, 1000],
              'dropout_rate':[0.2, 0.25, 0.5],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_