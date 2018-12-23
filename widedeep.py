# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 16:48:02 2018

@author: ARAVI
"""

import torch
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np

xy=np.loadtxt('data/diabetes.csv',delimiter=',',dtype=np.float32)

x=Variable(torch.from_numpy(xy[:,0:-1]))
y=Variable(torch.from_numpy(xy[:,[-1]]))

#print(x.data.shape[1])
#print(y.data.shape)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.l1=torch.nn.Linear(x.data.shape[1],6)
        self.l2=torch.nn.Linear(6,5)
        self.l3=torch.nn.Linear(5,y.data.shape[1])
        
    def forward(self,x):
        out1=self.l1(x)
        out2=f.sigmoid(self.l2(out1))
        y_pred=f.relu(self.l3(out2))
        return y_pred
    
model= Model()

criterion=torch.nn.BCELoss(size_average=True)
optimizer =torch.optim.SGD(model.parameters(),lr=0.1)

for epoch in range(100):
    y_pred=model(x)
    loss=criterion(y_pred,y)
    print(epoch,loss.data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
testval=np.array([-0.294118,0.487437,0.180328,-0.292929,0,0.00149028,-0.53117,-0.0333333],dtype=np.float32)
test=Variable(torch.from_numpy(testval))
#print(test)
print("value is",model(test).data)
    

    
    