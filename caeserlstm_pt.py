# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 16:51:32 2018

@author: asesagiri
"""

import torch
import caesercipher
import lstmdataset
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as f
import numpy as np

adataset=lstmdataset.getdataset(100,10)




hidden_dim=10#hidden_dim
batch_size=1#batch_size
embedding_dim=10#embedding_size
vocab_size=len(caesercipher.vocab)

def zero_hidden():
        return (torch.zeros(1,1,hidden_dim),torch.zeros(1,1,hidden_dim))

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.embed=nn.Embedding(vocab_size,embedding_dim)
        self.lstm=nn.LSTM(embedding_dim,hidden_dim)
        self.linear=nn.Linear(hidden_dim,vocab_size)
        
            
    def forward(self,x):
        out1=self.embed(x)
        out2,out3=self.lstm(out1.unsqueeze(1),zero_hidden())
        out4=self.linear(out2)
        return f.softmax(out4)
        #return out4
    
model=Model()

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)


def train(epoch):
    for original,encrypted in adataset:
        #print(epoch, len(original),len(encrypted))
        optimizer.zero_grad()
        output=model(original)
        output=output.transpose(1,2)
        encrypted=encrypted.unsqueeze(1)
        loss=criterion(output,encrypted)
        loss.backward()
        optimizer.step()
    print(epoch, loss.item())
        
for epoch in range(100):
    train(epoch)

#print(model.parameters())

#testing
testtext='i am godda'
testip=caesercipher.encryptedindex(testtext)[0]
#_,op=f.softmax(model(testip)).max(dim=2)
_,op=model(testip).max(dim=2)
op=op.squeeze(1)
print("predicted is",[caesercipher.vocab[x] for x in op])