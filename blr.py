# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 16:57:04 2018

@author: asesagiri
"""

import numpy as np
import matplotlib.pyplot as plt

x1_data=[1.0,2.0,3.0]
x2_data=[10.0,12.0,13.0]
y_data=[20.0,41.0,56.0]

#most basic model
def forward(x1,x2):
    return (x1*w1) + (x2*w2)

#loss function
    
def loss(x1,x2,y):
    y_pred=forward(x1,x2)
    return (y_pred-y)*(y_pred-y)

w_list=[]
mse_list=[]

for w1 in np.arange(0.0,4.0,0.1):
    print("w=",w1)
    w2=w1
    l_sum=0
    for x1_val,x2_val,y_val in zip(x1_data,x2_data,y_data):
        y_pred_val=forward(x1_val,x2_val)
        l=loss(x1_val,x2_val,y_val)
        l_sum=l_sum+l
        print("\t",x1_val,x2_val,y_val,y_pred_val,l)
    print("MSE",l_sum/len(x1_data))
    w_list.append(w1)
    mse_list.append(l_sum/len(x1_data))
    
plt.plot(w_list,mse_list)
plt.xlabel("w")
plt.ylabel("Loss")
plt.show()

    
        






