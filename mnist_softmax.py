# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 11:07:56 2018

@author: asesagiri
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional  as f
from torch.autograd import Variable
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np

batch_size=64

train_set=datasets.MNIST(root='./data/mnist/',train=True,transform=transforms.ToTensor(),download=True)
test_set=datasets.MNIST(root='./data/mnist/',train=False,transform=transforms.ToTensor())

img=torchvision.utils.make_grid(test_set[0][0])
#img = img / 2 + 0.5     # unnormalize
#npimg = img.numpy()
fig=plt.figure()
#plt.imshow(npimg)
plt.imshow(np.transpose(img, (1, 2, 0)))
plt.show()

train_loader=torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_set)


examples=enumerate(test_loader)
batch_id,(example_data,labels)=next(examples)
print(example_data.shape)


fig=plt.figure()
plt.imshow(example_data[1][0])
fig


for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0])
    plt.title(labels[i])
    plt.xticks([])
    plt.yticks([])
    
fig

class Model(nn.Module):
    
    def __init__(self):
        super(Model,self).__init__()
        self.l1=nn.Linear(784,600)
        self.l2=nn.Linear(600,300)
        self.l3=nn.Linear(300,100)
        self.l4=nn.Linear(100,10)
        
    def forward(self,x):
        x1=x.view(-1,784)
        out1=f.relu(self.l1(x1))
        out2=f.relu(self.l2(out1))
        out3=f.relu(self.l3(out2))
        return self.l4(out3)

model=Model()
    
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        data,target=Variable(data), Variable(target)
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
    for data,target in test_loader:
        data,target=Variable(data), Variable(target)
        output=model(data)
        test_loss+=criterion(output,target)
        pred=output.data.max(1,keepdim=True)[1]
        correct+=pred.eq(target.data.view_as(pred)).cpu().sum()
        
    test_loss/=len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        
    
for epoch in range(1,2):
    train(epoch)
    test(epoch)
        
output_new=model(example_data)

fig=plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0])
    plt.title(output_new.data.max(1,keepdim=True)[1][i].item())
    plt.xticks([])
    plt.yticks([])
    
fig

        
        