#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:24:12 2020

@author: kumet
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
#training Apriori on the dataset
from apyori import apriori
support = 3*7/7500 #support of a product that is purchased 3 times a day
rules = apriori(transactions,min_support = support,min_confidence = 0.2, min_lift = 3, min_length = 2)

#visualising the results
results = list(rules) #their relavence is sorted by a combination of support, lift, confidence by the apyori class.

