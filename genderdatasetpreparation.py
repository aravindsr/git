# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 21:08:47 2018

@author: ARAVI
"""

#the task is to create one csv file with two columns - name and gender
#the inputs are multiple text files

import os
import glob
import numpy as np

filepath="./data/gender/names/"
fileslist=os.listdir(filepath)
print(len(fileslist))

names=[]
genders=[]

for file in glob.glob(os.path.join(filepath,'*.txt')):
    xy=np.loadtxt(file,delimiter=',',dtype=str)
    for i in range(len(xy)):
        names.append(xy[i,0])
        genders.append(xy[i,1])
        #print(xy[i,0],xy[i,1])
    
fulldata=(names,genders)
np.savetxt('gendersdb.csv',np.column_stack((names,genders)),fmt='%s',delimiter=",")
