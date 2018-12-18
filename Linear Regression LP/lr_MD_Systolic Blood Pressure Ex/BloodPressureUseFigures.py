# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:27:58 2018

@author: philt
"""
# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Injecting noise to see the effect on r-sq'ed
#df = pd.read_excel('mlr02wAddedNoise.xlsx')
df = pd.read_excel('mlr02.xlsx')
X = df.as_matrix()

print('X ', X)

fig = plt.figure()
title = fig.suptitle("Verify Systolic Data")
fig.subplots_adjust(top=0.85, wspace=0.3)

ax1 = fig.add_subplot(121)
ax1.set_xlabel("Age in Years")
ax1.set_ylabel("Blood Pressue")
ax1.scatter(X[:, 1], X[:, 0])

ax2 = fig.add_subplot(122)
ax2.set_xlabel("Weight in Pounds")
ax2.set_ylabel("Blood Pressue")
ax2.scatter(X[:, 2], X[:, 0])
#fig = plt.figure(figsize = (6, 4))

#plt.scatter(X[:, 1], X[:, 0])
#plt.show()
#
#plt.scatter(X[:, 2], X[:, 0])
#plt.show()

# Bias
df['ones'] = 1
Y = df['X1']

# Injecting noise to see the effect on r-sq
#X = df[['X2', 'X3', 'Noise', 'ones']]
X = df[['X2', 'X3', 'ones']]
#print(X)
#print(Y)

X2only = df[['X2', 'ones']]
X3only = df[['X3', 'ones']]

def get_r2(X, Y):
    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    Yhat = X.dot(w)
    
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    return r2

print("r2 for x2 only:", get_r2(X2only ,Y))
print("r2 for x3 only:", get_r2(X3only, Y))
print("r2 for both:", get_r2(X, Y))
    