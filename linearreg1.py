# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 11:50:43 2018

@author: ARAVI
"""

import torch
from torch.autograd import Variable

x1=Variable(torch.tensor([[1],[2],[3]]))
x2=Variable(torch.tensor([[3],[5],[5]]))
y=Variable(torch.tensor([[11],[19],[24]]))

class lnreg(torch.nn.Module):
    
    def __init__(self):
        super(lnreg,self).__init__()
        self.linear=torch.nn.Linear(2,1)
        
    def forward(self,x1,x2):
        y_pred=self.linear(x1,x2)
        return y_pred
    
    
ln1=lnreg()

criterion=torch.nn.MSELoss(size_average=False)
optimizer1=torch.optim.SGD(ln1.parameters(),lr=0.01)


for epoch in range(10):
    y_pred=ln1(x1,x2)
    loss=criterion(y_pred,y)
    print(epoch,loss.data[0])
    optimizer1.zero_grad()    
    loss.backward()
    optimizer1.step()
    
    
hours_lala1=Variable(torch.tensor([4]))
hours_lala2=Variable(torch.tensor([7]))
y_pred=ln1((hours_lala1,hours_lala2))
print("Output is",ln1(hours_lala1,hours_lala2).data[0][0])
        