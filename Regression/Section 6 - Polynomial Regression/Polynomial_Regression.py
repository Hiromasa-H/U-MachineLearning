#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:06:01 2020

@author: kumet
"""


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
"""
#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)

#Feature Scaling is done by the library
"""

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


#Visualising the Polynomial Regression results
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color="red")
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color="blue")
plt.title("truth or bluff (Polynomial regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show


#Visualising the Linear Regression results
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg.predict(X),color="blue")
plt.title("truth or bluff (Linear regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show


#Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])
#Prediciting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))