# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:27:37 2018

@author: asesagiri
"""

import torchvision
import torch
import numpy


data_path = './data/catsndogs_v2/'
train_dataset = torchvision.datasets.ImageFolder(root=data_path,transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1,num_workers=0,shuffle=True)

for i,(data,labels) in enumerate(train_loader):
    print(data.shape)

    
