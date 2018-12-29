# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:27:37 2018

@author: asesagiri
"""

import torchvision
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from torchvision import transforms
import PIL.Image as Image

data_path = './data/catsndogs_v2/'
#train_dataset = torchvision.datasets.ImageFolder(root=data_path,transform=torchvision.transforms.ToTensor())
train_dataset = torchvision.datasets.ImageFolder(root=data_path,
                                                 transform=transforms.Compose([
                                                 transforms.CenterCrop(300),
                                                 transforms.Grayscale(num_output_channels=1),
                                                 transforms.ToTensor(),
                                                 #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 ]))
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=100,num_workers=0,shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.l1=nn.Conv2d(1,20,kernel_size=3)
        self.l2=nn.Conv2d(20,30,kernel_size=3)
        self.mp1=nn.MaxPool2d(2)
        self.lc=nn.Linear(159870,2)
        
    def forward(self,x):
        in_size=x.size(0)
        #test=x.view(in_size,-1)
        #print(x.size(),in_size)
        out1=f.relu(self.mp1(self.l1(x)))
        out2=f.relu(self.mp1(self.l2(out1)))
        out2=out2.view(in_size,-1)
        out3=self.lc(out2)
        #return f.log_softmax(out3) #use this only if you are using crossentropy loss
        return out3
    
model=Model()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)#,momentum=0.9



for i,(data,labels) in enumerate(train_loader,0):
    model.train()
    data,labels=Variable(data),Variable(labels)
    optimizer.zero_grad()
    output=model(data)
    pred=output.data.max(1,keepdim=True)[1]
    #print(labels,pred)
    loss=criterion(output,labels)
    loss.backward()
    optimizer.step()
    print(i,loss.data)

total=0
correct=0
with torch.no_grad():
    model.eval()
    for data in train_loader:
        images,labels=data
        output=model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct+=(predicted==labels).sum().item()
    
print('Accuracy of the network on the images: %d %%' % (
    100 * correct / total))
        



torch.save(model.state_dict(),'./saved_models/catsndogs_4k_nonormali_adam.pth')


preprocess = transforms.Compose([
transforms.Grayscale(num_output_channels=1),
transforms.CenterCrop(300),
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
    
#testing the prediction
image=Image.open('./data/catsndogs/train/dog.4903.jpg')
image = image.convert('RGB')
img_tensor=preprocess(image)
img2=img_tensor.unsqueeze(-4)
print(model(img2).data)
