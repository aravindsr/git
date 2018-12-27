# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 13:43:47 2018

@author: asesagiri
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import torch
import numpy as np

X,Y=make_regression(n_samples=200,n_features=10,noise=10)
print(X.shape,Y.shape)
#fig,ax=plt.subplots()
#ax.plot(X,Y,"bo")

x=torch.from_numpy(X).float()
print(x.shape)
y1=torch.from_numpy(Y.reshape((200,1))).float()
print(Y,y1)