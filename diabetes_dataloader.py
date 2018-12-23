# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 20:32:33 2018

@author: ARAVI
"""

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as f
from torch.utils.data import Dataset, DataLoader


class diabetesdataset(Dataset):
    def __init__(self):
        xy=np.loadtxt('./data/diabetes.csv',delimiter=',',dtype=np.float32)
        self.len=xy.shape[0]
        self.x=torch.from_numpy(xy[:,0:-1])
        self.y=torch.from_numpy(xy[:,-1])
        
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.len
    
print('yes')    
dataset=diabetesdataset()

#splitting the data into train and test datasets
train_size=int(0.8*len(dataset))
test_size=len(dataset)-train_size
print(train_size,test_size,len(dataset))
trainset,testset=torch.utils.data.random_split(dataset,[train_size,test_size])


train_loader=DataLoader(dataset=trainset,batch_size=32,shuffle=True,num_workers=0)
test_loader=DataLoader(dataset=testset,batch_size=32,shuffle=True,num_workers=0)

#print("test size is ",test_loader.len)

print('ok')

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.l1=torch.nn.Linear(8,6)
        self.l2=torch.nn.Linear(6,4)
        self.l3=torch.nn.Linear(4,1)
    
    def forward(self,x):
        out1=f.sigmoid(self.l1(x))
        out2=f.sigmoid(self.l2(out1))
        y_pred=f.sigmoid(self.l3(out2))
        return y_pred
    
model=Model()

criterion=torch.nn.BCELoss(size_average=True)
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

#training
for epoch in range(2):
    for i,data in enumerate(train_loader,0):
        #get the data from the dataloders
        inputs,labels=data
        inputs,labels=Variable(inputs),Variable(labels)
        
        #forward
        y_pred=model(inputs)
        #loss
        loss=criterion(y_pred,labels)
        #print(epoch,i,loss.data)
        #backward and step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
testval=np.array([-0.294118,0.487437,0.180328,-0.292929,0,0.00149028,-0.53117,-0.0333333],dtype=np.float32)
test=Variable(torch.from_numpy(testval))
#print(test)
print("value is",model(test).data>0.5)
        
#testing
for i,data in enumerate(test_loader,0):
    inputs,labels=data
    y_pred=model(inputs)
    loss=criterion(y_pred,labels)
    print(i,loss.data)
     
     



        

        