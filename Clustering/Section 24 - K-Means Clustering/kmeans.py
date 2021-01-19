#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:20:19 2020

@author: kumet
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import ing the dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

#using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
     kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10)
     kmeans.fit(X)
     wcss.append(kmeans.inertia_)
     
plt.plot(range(1,11),wcss)
plt.title("the elbow method")
plt.xlabel("number of clusters")
plt.ylabel("WCSS")
plt.show()

#Applying k-means to the dataset
kmeans = KMeans(n_clusters = 5, init="k-means++", max_iter = 300, n_init = 10)
y_kmeans = kmeans.fit_predict(X)

#visualizing the clusters
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0, 1], s=100, c="red", label ="cluster 1")
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1, 1], s=100, c="blue", label ="cluster 2")
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2, 1], s=100, c="green", label ="cluster 3")
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3, 1], s=100, c="cyan", label ="cluster 4")
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4, 1], s=100, c="magenta", label ="cluster 5")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300,c ="yellow", label ="centroids")
plt.title("clusters of clients")
plt.xlabel("Annual income")
plt.ylabel("spending score (1-100)")
plt.legend()
plt.show()