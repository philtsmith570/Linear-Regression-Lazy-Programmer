# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 09:12:20 2018

@author: philt
"""

import numpy as np
import matplotlib.pyplot as plt

#Make some data
N = 100
X= np.linspace(0, 6*np.pi, N)
Y = np.sin(X)

plt.plot(X, Y)
plt.show()

def make_poly(X, deg):
    n = len(X)
    data = [np.ones(n)]
    for d in xrange(deg):
        data.append(X**(d+1))
    return np.vstack(data).T

def fit(X, Y):
    return np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    

def fit_and_display(X, Y, sample, deg):
    N = len(X)
    train_idx = np.random.choice(N, sample)
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]
    
    plt.scatter(Xtrain, Ytrain)
    plt.show()
    
    #fit polynomial
    Xtrain_poly = make_poly(Xtrain, deg)
    w = fit(Xtrain_poly, Ytrain)
    
    #Disply polynomial
    X_poly = make_poly(X, deg)
    Y_hat = X_poly.dot(w)
    plt.plot(X, Y)
    plt.plot(X, Y_hat)
    plt.scatter(Xtrain, Ytrain)
    plt.title("deg = ", deg)
    plt.show()
    
for deg in (5, 6, 7, 8, 9):
    fit_and_display(X, Y, 10, deg)

def get_mse(Y, Yhat):
    d= Y - Yhat
    return d.dot(d) / len(d)

def plot_train_vs_test_curves(X, Y, sample=20, max_deg=20):
    N = len(N)
    train_idx = np.random.choice(N, sample)
    Xtrain = X[train_idx]
    Ytrain = Y(train_idx)
    
    test_idx = [idx for idx in xrange(N) if idx not in train_idx]
    #test_idx = np.random.choice(N, sample)
    Xtest = X[test_idx]
    Ytest = Y[test_idx]
    
    mse_trains = []
    mse_tests = []
    
    for deg in xrange(max_deg + 1):
        Xtrain_poly = make_poly(Xtrain, deg)
        w = fit(Xtrain_poly, Ytrain)
        Yhat_train = Xtrain_poly.dot(w)
        
    
    