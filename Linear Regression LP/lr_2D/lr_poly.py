# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:05:43 2018

@author: philt
"""


import numpy as np
import matplotlib.pyplot as plt


# load the data
X = []
Y = []
for line in open('data_poly.csv'):
    x, y = line.split(',')
    x = float(x)
    
    X.append([1, x, x*x]) # add the bias term
    Y.append(float(y))

# let's turn X and Y into numpy arrays since that will be useful later
X = np.array(X)
Y = np.array(Y)


# let's plot the data to see what it looks like
plt.scatter(X[:,1], Y)
plt.show()


# apply the equations we learned to calculate a and b
# numpy has a special method for solving Ax = b
# so we don't use x = inv(A)*b
# note: the * operator does element-by-element multiplication in numpy
#       np.dot() does what we expect for matrix multiplication
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# Plot the results
plt.scatter(X[:,1], Yhat)
plt.plot(sorted(X[:,1]), sorted(Yhat))
plt.show()


# determine how good the model is by computing the r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)
