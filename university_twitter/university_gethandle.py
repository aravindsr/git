# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:25:35 2019

@author: asesagiri
"""

import tweepy
import numpy as np
import pandas as pd

#Twitter authentication
#first one


api = tweepy.API(auth,wait_on_rate_limit=True)

#reading data from file
univdata = pd.read_csv('universitylist_4.txt', header=None,delimiter=None,dtype=object)
print(len(univdata))
#print(univdata[0][1])
#print(univdata.shape)

op = pd.DataFrame([])

f = open('op1000_4.txt','a',encoding="utf-8")
f.write('id'+'\t'+'handle'+'\t'+'name'+'\t'+'location'+'\t'+'followers_count'+'\t'+'friends_count'+'\t'+'tweets_count'+'\t'+'url'+'\t'+'description')

for i in range(2000):
    print(univdata[0][i])
    univs = api.search_users(univdata[0][i])
    univslen=len(univs)
    print('Retrieved list of users is',univslen,univdata[0][i])
    if(univslen>0):
        #print(str(univs[0].id)+'\t'+str(univs[0].screen_name)+'\t'+str(univs[0].name)+'\t'+str(univs[0].location)+'\t'+str(univs[0].followers_count)+'\t'+str(univs[0].friends_count)+'\t'+str(univs[0].url)+'\t'+univs[0].description)
        f.write('\n' + str(univs[0].id)+'\t'+str(univs[0].screen_name)+'\t'+str(univs[0].name)+'\t'+str(univs[0].location)+'\t'+str(univs[0].followers_count)+'\t'+str(univs[0].friends_count)+'\t'+str(univs[0].statuses_count)+'\t'+str(univs[0].url)+'\t'+univs[0].description)
        f.flush()
f.close()       
