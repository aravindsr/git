# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 12:57:21 2018

@author: asesagiri
"""

import requests
import json
import numpy as np
import pandas as pd


doidata = pd.read_csv('universitylist_4.txt', header=None,delimiter=None,dtype=object)
print(len(doidata))
#print(univdata[0][1])
#print(univdata.shape)

op = pd.DataFrame([])

f = open('opdoi_1.txt','a',encoding="utf-8")

#   The credentials to be used
login = {
    'username': 'asesagiri@ntu.edu.sg',
    'password': 'player99'
}

#   Send credentials to login url to retrieve token. Raise
#   an error, if the return code indicates a problem.
#   Please use the URL of the system you'd like to access the API
#   in the example below.
resp = requests.post('https://app.dimensions.ai/api/auth.json', json=login)
resp.raise_for_status()

#   Create http header using the generated token.
headers = {
    'Authorization': "JWT " + resp.json()['token']
}


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




#print(headers)
#   Execute DSL query.
resp = requests.post(
    'https://app.dimensions.ai/api/dsl.json',
   #data='search publications where doi=\"10.1007/s13142-015-0313-4\" return publications [FOR]',
   data='search publications where doi=\"10.1007/s13142-015-0313-4\" return publications',
   #data='search publications  for \"health coaching\" return publications',
    headers=headers)
result=resp.json()
result_new=json.dumps(result)
#result_new=result_new.replace('\'','"')
#result_new=result_new.replace('None','"None"')
#   Display raw result
print(result_new)
