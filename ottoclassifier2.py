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

#convert output variable to factor
trainxy.iloc[:,-1:]=trainxy.iloc[:,-1:].apply(lambda x: pd.factorize(x)[0])

#deleting first column which is row number
trainxy=trainxy.drop(trainxy.columns[[0]], axis=1)

#converting dataframe to numpy array
data_set=trainxy.values
data_set=data_set.astype(np.float32)


#splitting numpy array to train and test sets
train_size=int(0.8*len(data_set))
test_size=len(data_set)-train_size
train_set,test_set=torch.utils.data.random_split(data_set,[train_size,test_size])
print(len(train_set),len(test_set))



#creating training and test dataloaders
train_loader=DataLoader(dataset=train_set,batch_size=80,shuffle=False)
test_loader=DataLoader(dataset=test_set,batch_size=80,shuffle=False)

classes = ('class1', 'class2', 'class3', 'class4','class5', 'class6', 'class7', 'class8', 'class9')

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
        #out4=f.relu(self.l4(out3))
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
    model.eval()
    test_loss=0
    correct=0
    for batch_idx,fulldata in enumerate(test_loader,0):
        data=fulldata[:,0:-1]
        target=fulldata[:,-1]
        data,target=Variable(data),Variable(torch.tensor(target,dtype=torch.long))
        output=model(data)
        test_loss+=criterion(output,target)
        pred=output.data.max(1,keepdim=True)[1]
        correct+=pred.eq(target.data.view_as(pred)).cpu().sum()
        
            
    test_loss /=len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        

def classtest():
    class_correct = list(0. for i in range(9))
    class_total = list(0. for i in range(9))
    with torch.no_grad():
        counter=0;
        for fulldata in train_loader:
            data=fulldata[:,0:-1]
            labels=fulldata[:,-1]
            data,labels=Variable(data),Variable(torch.tensor(labels,dtype=torch.long))
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            counter+=1
            #print(counter,labels)
            for i in range(9):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                

    for i in range(9):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        
        
for epoch in range(1):
       train(epoch)
       #test(epoch)

''''     
exampletest=test_set[50,][0:-1]
exampleresult=test_set[50,][-1]
print(exampletest,exampleresult)
ex=Variable(torch.tensor(exampletest,dtype=torch.float32))
op=model(ex)
print(op.data,torch.max(op,0))
'''
classtest()