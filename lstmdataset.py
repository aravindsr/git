# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 14:38:35 2018

@author: asesagiri
"""
import torch
import random
import caesercipher

def getdataset(num_examples,message_length):
    dataset=[]
    for x in range(num_examples):
        normaltext=''.join([random.choice(caesercipher.vocab) for x in range(message_length)])
        encrypted=caesercipher.encrypt(''.join(normaltext))
        n_index =[caesercipher.vocab.index(x) for x in normaltext]
        e_index =[caesercipher.vocab.index(x) for x in encrypted]
        dataset.append([torch.tensor(n_index),torch.tensor(e_index)])
    return dataset
    
        