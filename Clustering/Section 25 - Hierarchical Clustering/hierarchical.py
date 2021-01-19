#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 00:04:06 2020

@author: kumet
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset with pd
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

# finding the optimal number of clusters with the dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method = "ward"))
plt.title("dendrogram")
plt.xlabel("customers")
plt.ylabel("euclidean distances")
plt.show()

#fitting hc to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean",linkage = "ward")
y_hc = hc.fit_predict(X)

# visualizing the clusters
y_kmeans = y_hc
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0, 1], s=100, c="red", label ="cluster 1")
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1, 1], s=100, c="blue", label ="cluster 2")
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2, 1], s=100, c="green", label ="cluster 3")
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3, 1], s=100, c="cyan", label ="cluster 4")
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4, 1], s=100, c="magenta", label ="cluster 5")
#plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300,c ="yellow", label ="centroids")
plt.title("clusters of clients")
plt.xlabel("Annual income")
plt.ylabel("spending score (1-100)")
plt.legend()
plt.show()