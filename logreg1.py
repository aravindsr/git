# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 17:57:16 2018

@author: ARAVI
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as f

x=Variable(torch.tensor([[1,3],[2,5],[3,6]],dtype=torch.float32))
y=Variable(torch.tensor([[0],[0],[1]],dtype=torch.float32))

class logreg (torch.nn.Module):
    def __init__(self):
        super(logreg,self).__init__()
        self.logistic=torch.nn.Linear(2,1)
    
    def forward(self,x):
        y_pred=f.sigmoid(self.logistic(x))
        return y_pred
        
lg1=logreg()

criterion=torch.nn.BCELoss(size_average=True)
optimizer=torch.optim.SGD(lg1.parameters(),lr=0.01)

for epoch in range(500):
    #do the forward pass
    y_pred=lg1(x)
    
    #do the loss
    loss=criterion(y_pred,y)
    print(epoch,loss.data)
    
    #do the backward
    optimizer.zero_grad()
    loss.backward
    
    #do the step
    optimizer.step()
    
hours_pred=Variable(torch.tensor([[2,3]],dtype=torch.float32))
print("the output is",lg1(hours_pred).data[0][0]>0.5)
        
        
