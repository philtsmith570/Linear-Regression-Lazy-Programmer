# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 09:17:02 2017

@author: philt
"""

import numpy as np
import matplotlib.pyplot as plt

# load data
X = []
Y = []

# Can use cv reader  - Try later
for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# Turn the input into numpy arrays
X = np.array(X)
Y = np.array(Y)

plt.scatter(X, Y)
plt.show()

# Apply eqns for linear regression  a,b
denominator = X.dot(X) - X.mean()*X.sum()

a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

# Predicted Y
Yhat = a*X + b
plt.scatter(X,Y)
plt.plot(X, Yhat)
plt.show()

# Calculate R-Squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("The R2 is :  ", r2)