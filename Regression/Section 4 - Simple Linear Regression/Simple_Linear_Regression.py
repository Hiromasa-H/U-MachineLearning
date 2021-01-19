#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:38:12 2020

@author: kumet
"""
#simple linear regression

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3,random_state = 0)

#Feature Scaling will be taken care of via the library

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#visualizing the training set results
plt.scatter(X_train,y_train, color="red")
plt.plot(X_train,regressor.predict(X_train), color="blue")
plt.title("salary vs experience (Training set)")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()

#visualizing the testing set results
plt.scatter(X_test,y_test, color="red")
plt.plot(X_train,regressor.predict(X_train), color="blue")
plt.title("salary vs experience (Testing set)")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()