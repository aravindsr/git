# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 12:41:55 2018

@author: ARAVI
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as f
from torchvision import datasets,transforms
import torch.utils.data.dataloader as dataloader #come back here

batch_size=64

train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1=nn.Conv2d(1,10,kernel_size=5)
        self.conv2=nn.Conv2d(10,20,kernel_size=5)
        self.mp=nn.MaxPool2d(2)
        self.fc=nn.Linear(320,10)
        
    def forward(self,x):
        in_size=x.size(0)
        print(x,x.size(),in_size)
        out1=f.relu(self.mp(self.conv1(x)))
        out2=f.relu(self.mp(self.conv2(out1)))
        out3=out2.view(in_size,-1) #flattening
        out4=self.fc(out3)
        return f.log_softmax(out4)
    
model=Model()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader,0):
        data,target=Variable(data),Variable(target)
        optimizer.zero_grad()
        output=model(data)
        loss=f.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))
            
            
def test():
    model.eval()
    test_loss=0
    correct=0
    for data,target in test_loader:
        data,target=Variable(data,volatile=True),Variable(target)
        output=model(data)
        test_loss+=f.nll_loss(output,target)
        pred=output.data.max(1,keepdim=True)[1]
        correct+=pred.eq(target.data.view_as(pred)).cpu().sum()
        
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        

for epoch in range(1,2):
    train(epoch)
    #test()
        

    
    
    