# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 23:54:19 2018

@author: ARAVI
"""

import torch
from torch.autograd import Variable

w1=Variable(torch.tensor([1.0]),requires_grad=True)
w2=Variable(torch.tensor([1.0]),requires_grad=True)

x1=[1,2,3]
x2=[3,5,6]
y=[11,19,24]

def forward(x1,x2):
    #print(w)
    return (x1*w1)+(x2*w2)

def loss(x1,x2,y):
    y_predicted_value=forward(x1,x2)
    return (y_predicted_value-y)*(y_predicted_value-y)

print("Before Training",forward(4,7).data[0])


for epoch in range(10):
    for x1_val,x2_val,y_val in zip(x1,x2,y):
        l=loss(x1_val,x2_val,y_val)
        l.backward()
        print("\tgrad",x1_val,x2_val,y_val,w1.grad.data[0],w2.grad.data)
        w1.data=w1.data-0.01*w1.grad.data
        w2.data=w2.data-0.01*w2.grad.data
    
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        
    print("progress", epoch, l.data[0])
    

print("after training", forward(4,7).data[0])    

    

    