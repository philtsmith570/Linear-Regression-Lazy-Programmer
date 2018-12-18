# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:18:44 2018

@author: philt
"""

import numpy as np

X = np.linspace(0, 6*np.pi, 10)
n = len(X)
print("n = ",n)
data = [np.ones(n)]
#print("Data = ", data)
for d in range(2):
    data.append(X**(d+1))
    print("X: ", X )
    print("Data append", data)

for n in range(4):
    X.append(n)
print(X)