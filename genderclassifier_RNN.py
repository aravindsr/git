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
import pickle
import torch.nn as nn
from torch.autograd import Variable

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'Manager':
            from settings import Manager
            return Manager
        return super().find_class(module, name)


'''
text='hH'
ar=[]
for char in text:
    print(ord(char))
    ar.append(ord(char))
   

print(ar)


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
opfile=open('./data/gender/genderdb.txt','wb')
pickle.dump(dataset,opfile)
opfile.close()

'''

#dataset already created earlier, will be loaded
#ip=open('./data/gender/genderdb.txt','rb')
#dataset=pickle.load(ip)
#ip.close()

dataset = CustomUnpickler(open('./data/gender/genderdb.txt', 'rb')).load()
print(len(dataset))



#splitting the dataset into train and test sets
train_size=int(0.80*len(dataset))
test_size=len(dataset)-train_size
train_set,test_set= torch.utils.data.random_split(dataset,[train_size,test_size])
print(len(train_set),len(test_set))

#creation of data loaders
train_loader=DataLoader(dataset=train_set,shuffle=True,batch_size=1000)
test_loader=DataLoader(dataset=test_set,shuffle=False,batch_size=1000)

n_steps = 28
n_inputs = 20
n_neurons = 150
n_outputs = 2
n_epochs = 2

class Model(nn.Module):
    def __init__(self,batch_size, n_steps, n_inputs, n_neurons, n_outputs):
        super(Model,self).__init__()
        self.n_steps=n_steps
        self.batch_size=1000
        self.n_inputs=n_inputs
        self.n_neurons=n_neurons
        self.n_outputs=n_outputs
        self.rnn=nn.RNN(self.n_inputs,self.n_neurons)
        self.lc=nn.Linear(self.n_neurons,self.n_outputs)
        
    def init_hidden(self,):
        return (torch.zeros(1,self.batch_size,self.n_neurons))
        
    def forward(self,x):
        
        x = x.permute(1, 0, 2) 
        self.batch_size = x.size(1)
        self.hidden = self.init_hidden()
        rnn_out, self.hidden = self.rnn(x, self.hidden)      
        out = self.lc(self.hidden)
        return out.view(-1, self.n_outputs) # batch_size X n_output
      
'''
#casual testing     
dataiter = iter(train_loader)
data, labels = dataiter.next()
data,labels=data.type(torch.float32),labels.type(torch.float32)
#model = ImageRNN(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS)
model = Model(1000, 28, 20, 15, 2)
logits = model(data)
print(logits.data.max(),labels.shape)
'''
   
model=Model(1000,28,20,15,2)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)     
        
for batch_idx,(data,labels) in enumerate(train_loader,0):
    model.train()
    optimizer.zero_grad()
    model.hidden = model.init_hidden()
    data,labels=data.type(torch.float32),labels.type(torch.LongTensor)
    labels=labels.squeeze(-1)
    output=model(data)
    pred=output.data.max(1,keepdim=True)[1]
    #print(torch.max(labels, 1)[1])
    loss=criterion(output,torch.max(labels, 1)[1])
    loss.backward()
    optimizer.step()
    print(batch_idx,loss.data)
 
torch.save(model.state_dict(),'./saved_models/Gender_RNN_model.pth')


correct = 0
total = 0
with torch.no_grad():
    for batch_idx,(data,labels) in enumerate(test_loader,0):
        data,labels=data.type(torch.float32),labels.type(torch.LongTensor)
        labels=labels.squeeze(-1)
        output=model(data)
        pred=output.data.max(1,keepdim=True)[1]
        total += labels.size(1)
        correct += (pred == torch.max(labels, 1)[1]).sum().item()
        print(batch_idx,loss.data)
    
print('Accuracy of the network on the 10000 test dataset: %d %%' % (
    100 * correct / total))
    



