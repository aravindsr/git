# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:55:51 2018

@author: asesagiri
"""
#load images
#converting images to tensors and create dataset
#split the dataset to train and test
#implement resnet

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import requests
import io


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Scale(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])


class cddataset(Dataset):
    def __init__(self):
       imagespath="./data/catsndogs/train/"
       imageslist=os.listdir(imagespath)
       print(len(imageslist))
       catcount=0
       dogcount=0
       for file in imageslist:
           if file[:3]=="dog":
              dogcount+=1
           elif file[:3]=="cat":
               catcount+=1
               
               
               
       self.len=xy.shape[0]
       self.x=torch.from_numpy(xy[:,0:-1])
       self.y=torch.from_numpy(xy[:,-1])
        
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.len
 




image=Image.open("./data/catsndogs/train/cat.0.jpg")
#print(image.filename)
image = image.convert('RGB')
img_tensor = preprocess(image)
catlabel=torch.tensor([[1]])
doglabel=torch.tensor([[2]])

new1=torch.empty(5, 7, dtype=torch.float)
new1.x=torch.tensor(img_tensor)
new1.y=catlabel
print(new1)


'''
img_tensor = preprocess(image)
#print(img_tensor.shape)
img_tensor.unsqueeze_(0)
#print(img_tensor.shape)

testnp=np.asarray(image)
#plt.imshow(testnp)
#plt.show()
testen=torch.from_numpy(testnp)
trans1 = transforms.ToTensor()
testa=trans1(testnp)

print(img_tensor.shape)


#load images and get names
imagespath="./data/catsndogs/train/"
imageslist=os.listdir(imagespath)
print(len(imageslist))
catcount=0
dogcount=0
for file in imageslist:
    if file[:3]=="dog":
        dogcount+=1
        
        
    elif file[:3]=="cat":
        catcount+=1
        
        
print(dogcount,catcount)
'''