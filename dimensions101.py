# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 12:57:21 2018

@author: asesagiri
"""

import requests

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

#print(headers)
#   Execute DSL query.
resp = requests.post(
    'https://app.dimensions.ai/api/dsl.json',
   # data='search publications where doi=\"10.1007/s13142-015-0313-4\" return publications [FOR]',
   data='search publications  for \"artificial intelligence\" return publications',
    headers=headers)
result=resp.json()

#   Display raw result
#print(resp.json())
