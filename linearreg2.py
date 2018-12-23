# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 11:50:43 2018

@author: ARAVI
"""

import torch
from torch.autograd import Variable
import numpy as np


#x=np.array([[1.0,3.0],[2.0,5.0],[3.0,6.0]], dtype=np.float32)
#x1=torch.from_numpy(x)
 

x1=Variable(torch.tensor([[1.0,3.0],[2.0,5.0],[3.0,6.0]],dtype=torch.float32))
y=Variable(torch.tensor([[11.0],[19.0],[24.0]], dtype=torch.float32))

class lnreg(torch.nn.Module):
    
    def __init__(self):
        super(lnreg,self).__init__()
        self.linear=torch.nn.Linear(2,1)
        
    def forward(self,x1):
        y_pred=self.linear(x1)
        return y_pred
    
    
ln1=lnreg()

criterion=torch.nn.MSELoss(size_average=False)
optimizer1=torch.optim.SGD(ln1.parameters(),lr=0.01)


for epoch in range(5000):
    y_pred=ln1(x1)
    loss=criterion(y_pred,y)
    print(epoch,loss.data)
    optimizer1.zero_grad()    
    loss.backward()
    optimizer1.step()
    
    
hours_lala1=Variable(torch.tensor([[4.0,7.0]],dtype=torch.float32))

y_pred=ln1(hours_lala1)
print("Output is",ln1(hours_lala1).data[0][0])
        