# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 18:14:24 2018

@author: ARAVI
"""

import torchvision
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from torchvision import transforms,models
import PIL.Image as Image

data_path = './data/catsndogs_v3/'
#train_dataset = torchvision.datasets.ImageFolder(root=data_path,transform=torchvision.transforms.ToTensor())
train_dataset = torchvision.datasets.ImageFolder(root=data_path,
                                                 transform=transforms.Compose([
                                                 transforms.CenterCrop(224),
                                                 #transforms.Grayscale(num_output_channels=1),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ]))
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=100,num_workers=0,shuffle=True)

def get_model():
    premodel = models.resnet18(pretrained=True)#resnet
    premodel.fc = nn.Linear(512, 2)
    #premodel=models.densenet121(pretrained=True) #densenet
    #premodel.fc = nn.Linear(121, 2)
    return premodel

newmodel=get_model()

criterion= nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(newmodel.parameters(),lr=0.001)

total=0;
correct=0
for epoch in range(1):
    for data in train_loader:
        inputs,labels =data
        inputs,labels=Variable(inputs),Variable(labels)
        output=newmodel(inputs)
        _,op=torch.max(output,1)
        total += labels.size(0)
        print(total)
        correct+=(op==labels).sum().item()

print('Accuracy of the network on the images: %d %%' % (
    100 * correct / total))        

preprocess = transforms.Compose([
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
 std=[0.229, 0.224, 0.225])
])
    
#testing the prediction
image=Image.open('./data/catsndogs/train/dog.4903.jpg')
image = image.convert('RGB')
img_tensor=preprocess(image)
img2=img_tensor.unsqueeze(-4)
print(newmodel(img2).data)

    

