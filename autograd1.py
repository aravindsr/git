# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 23:54:19 2018

@author: ARAVI
"""

import torch
from torch.autograd import Variable

w=Variable(torch.tensor([1.0]),requires_grad=True)

x=[1,2,3]
y=[2,4,6]

def forward(x):
    print(w)
    return x*w

def loss(x,y):
    y_predicted_value=forward(x)
    return (y_predicted_value-y)*(y_predicted_value-y)

print("Before Training",forward(4).data[0])


for epoch in range(100):
    for x_val,y_val in zip(x,y):
        l=loss(x_val,y_val)
        l.backward()
        print("\tgrad",x_val,y_val,w.grad.data[0])
        w.data=w.data-0.01*w.grad.data
    
        w.grad.data.zero_()
        
    print("progress", epoch, l.data[0])
    

print("after training", forward(4).data[0])    

    

    