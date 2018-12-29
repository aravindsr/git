# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 20:32:18 2018

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
import ignite
import torchvision_imagefolder as lala
from ignite.metrics import (
    CategoricalAccuracy,
    Loss,
    Precision,
)
from ignite.engine import (
    create_supervised_evaluator,
    create_supervised_trainer,
    Events,
)

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
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=10,num_workers=0,shuffle=True)

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

trainer = create_supervised_trainer(model, optimizer, criterion)

trainer.run(train_loader, max_epochs=1)
