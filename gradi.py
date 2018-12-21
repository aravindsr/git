#utf-8 -*-
"""
Created on Fri Dec 21 16:57:04 2018

@author: asesagiri
"""

import numpy as np
import matplotlib.pyplot as plt

x1_data=[1.0,2.0,3.0]
x2_data=[2.0,3.0,4.0]
y_data=[8.0,13.0,18.0]
w1=1.0
w2=1.0

#most basic model
def forward(x1,x2):
    return (x1*w1) + (x2*w2)

#loss function
    
def loss(x1,x2,y):
    y_pred=forward(x1,x2)
    return (y_pred-y)*(y_pred-y)

#gradient for w1formula 2w(wx1+wx2-y)
def gradient1(x1,x2,y):
    return (2*x1)*((x1*w1)+(x2*w2)-y)
 
#gradient for w1
def gradient2(x1,x2,y):
    return (2*x2)*((x1*w1)+(x2*w2)-y)


print("Before training",4,forward(4,5))

w1_list=[]
w2_list=[]
loss_list=[]

for epoch in range(20):
    for x1_val,x2_val,y_val in zip(x1_data,x2_data,y_data):
        grad1=gradient1(x1_val,x2_val,y_val)
        grad2=gradient2(x1_val,x2_val,y_val)
        w1=w1-0.01*grad1
        w2=w2-0.01*grad2
        w1_list.append(w1)
        w2_list.append(w2)
        print("\tgrad: ", x1_val,x2_val,y_val,round(grad1,2),round(grad2,2))
        l=loss(x1_val,x2_val,y_val)
        loss_list.append(l)
        
    print("progress",epoch,"w1=",round(w1,2),"w2=",round(w2,2),"loss=",round(l,2))
        
print("After training",4,forward(4,5))       

plt.plot(w2_list,loss_list)
plt.xlabel("w1")
plt.ylabel("loss")
plt.show()

    
        






