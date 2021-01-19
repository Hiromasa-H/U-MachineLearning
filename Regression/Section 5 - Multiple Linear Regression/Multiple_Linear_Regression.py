#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:13:16 2020

@author: kumet
"""

#Multiple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable trap
X=X[:,1:]

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

#Feature Scaling is done by the library
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test score results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backwards Elimination
#preperation
#import statsmodels.formula.api as sm
#check lesson 49 for this
#X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
#X_opt = X[:,[0,1,2,3,4,5]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

#OLS method was different from the one used here
