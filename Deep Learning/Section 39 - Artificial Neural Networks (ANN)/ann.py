#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:52:45 2020

@author: kumet
"""

#-------------------preprocessing the data-------------------

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1]) #create dummy variables
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:13] #remove the 1st dummy variable to avoid the dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#-------------------Making the ANN-------------------
#import Keras libraries
import keras
from keras.models import Sequential #needed to in initialize the model
from keras.layers import Dense #needed to build the layers

#initialising the ANN
classifier = Sequential()
#we need 11 nodes and a rectefier activation function for the hidden layer, and a sigmoid function in the output layer
#learning rate will be set at


#adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init='uniform', activation = 'relu', input_dim=11))

#add a new hidden layer
classifier.add(Dense(output_dim = 6, init='uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 6, init='uniform', activation = 'relu'))

#add output layer
classifier.add(Dense(output_dim = 1, init='uniform', activation = 'sigmoid'))
#if there are more categories for the ouput use: 
#classifier.add(Dense(output_dim = 3, init='uniform', activation = 'softmax'))

#compiling the ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ["accuracy"])
#loss = categorical_crossentropy when you have more categories

#fitting the ANN to the Training set
#choose batch size and number of epochs
classifier.fit(X_train,y_train, batch_size = 10, nb_epoch = 100)

#-------------------Making the predictions and evaluating the model-------------------



# Fitting the Classifier to the Training set
# Create Classifier

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




#-------------------plotting the thing-------------------

#from keras.utils import plot_model
#plot_model(classifier, to_file='model.png')

history = classifier.fit(X_train,y_train, batch_size = 10, nb_epoch = 100)
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
#plt.plot(history.history['accuracy.value'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
