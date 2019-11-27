# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 14:18:49 2018

@author: asesagiri
"""
import torch

key=13
vocab=[char for char in 'abcdefghijklmnopqrstuvwxyz ']

def encrypt(text):
    indexes=[vocab.index(char) for char in text]
    newindexes=[(idx+key)%len(vocab) for idx in indexes]
    encrypted_chars=[vocab[idx] for idx in newindexes]
    encrypted_text=''.join(encrypted_chars)
    return encrypted_text

def encryptedindex(text):
    original=text
    encrypted=encrypt(text)
    orig_indexes =[vocab.index(x) for x in original]
    encr_indexes =[vocab.index(x) for x in encrypted]
    return ([torch.tensor(orig_indexes),torch.tensor(encr_indexes)])

print(encrypt('i am godda')) 
print(encryptedindex('i am godda')) 
