# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 21:37:09 2018

@author: ARAVI
"""

#So, basically the intention is to classify gender based on name using RNN
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as f
text='hH'

'''
ar=[]
for char in text:
    print(ord(char))
    ar.append(ord(char))
   

print(ar)
''' 

class genderdataset(Dataset):
    def __init__(self):
        xy=np.loadtxt('./data/gender/gendersdb.csv',delimiter=',',dtype=str)
        self.len=xy.shape[0]
        
        
        xvalues=np.zeros((self.len,1,20))
        yvalues=np.zeros((self.len,1,1))
        j=0
        for i in range(len(xy)):
            x_text=xy[i,0]
            y_text=xy[i,1]
            #print(x_text,y_text)
            for k in range(len(x_text)):
                xvalues[j][0][k]=ord(x_text[k])
            
            yvalues[j][0]=ord(y_text)
            j+=1
     
        self.x=xvalues
        self.y=yvalues
        
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.len
    
dataset=genderdataset()
print(len(dataset))


        