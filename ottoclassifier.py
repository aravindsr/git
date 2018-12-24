# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 15:06:25 2018

@author: asesagiri
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import torch.nn.functional as f


trainxy=pd.read_csv("./data/otto/train.csv")
testxy=pd.read_csv("./data/otto/test.csv")
#convert output variable to factor
trainxy.iloc[:,-1:]=trainxy.iloc[:,-1:].apply(lambda x: pd.factorize(x)[0])

#deleting first column which is row number
trainxy=trainxy.drop(trainxy.columns[[0]], axis=1)
testxy=testxy.drop(testxy.columns[[0]], axis=1)
#print(trainxy.tail(5))

#converting dataframe to numpy array
train_set=trainxy.values
test_set=testxy.values
train_set=train_set.astype(np.float32)
test_set=test_set.astype(np.float32)

'''
#splitting numpy array to train and test sets
train_size=int(0.8*len(xy))
test_size=len(xy)-train_size
train_set,test_set=torch.utils.data.random_split(xy,[train_size,test_size])
'''


#creating training and test dataloaders
train_loader=DataLoader(dataset=train_set,batch_size=100,shuffle=True)
test_loader=DataLoader(dataset=test_set,batch_size=100,shuffle=False)

#create the Model class
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.l1=nn.Linear(93,200)
        self.l2=nn.Linear(200,100)
        self.l3=nn.Linear(100,50)
        self.l4=nn.Linear(50,9)
        
    def forward(self,x):
        out1=f.relu(self.l1(x))
        out2=f.relu(self.l2(out1))
        out3=f.relu(self.l3(out2))
        return self.l4(out3)
    
model=Model()

#loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx,fulldata in enumerate(train_loader,0):
        data=fulldata[:,0:-1]
        target=fulldata[:,-1]
        data,target=Variable(data), Variable(torch.tensor(target,dtype=torch.long))
        optimizer.zero_grad()
        output=model(data)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))


def test(epoch):
    for batch_idx,fulldata in enumerate(test_loader,0):
        output=model(fulldata)
        print(output.data.max(1,keepdim=True)[1])
        
    

     
        
for epoch in range(1,2):
       train(epoch)
       test(epoch)
       
       