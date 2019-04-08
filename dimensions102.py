# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 12:57:21 2018

@author: asesagiri
"""

import requests
import json
import numpy as np
import pandas as pd



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

doidata = pd.read_csv('HWC_DOI.txt', header=None,delimiter=None,dtype=object)
print(len(doidata))

f = open('opdoifor.txt','a',encoding="utf-8")

for i in range(729):
    print(doidata[0][i])
    resp = requests.post(
    'https://app.dimensions.ai/api/dsl.json',
    data='search publications where doi=\"'+doidata[0][i]+'\" return publications [FOR]',
    #data='search publications where doi=\"'+doidata[0][i]+'\" return publications',
   #data='search publications  for \"health coaching\" return publications',
    headers=headers)
    result=resp.json()
    result_new=json.dumps(result)
    print(result_new)
    f.write('\n' + doidata[0][i]+'\t'+result_new)
    f.flush()

f.close()



#print(headers)
#   Execute DSL query.


#result_new=result_new.replace('\'','"')
#result_new=result_new.replace('None','"None"')
#   Display raw result

