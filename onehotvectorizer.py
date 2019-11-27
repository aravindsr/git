# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:47:03 2019

@author: ephsra
"""
from sklearn.feature_extraction.text import CountVectorizer

import seaborn as sns

corpus = ['aravind sesagiri raamkumar', 'aravind s raamkumar']

one_hot_vectorizer = CountVectorizer(binary=True,token_pattern = r"(?u)\b\w+\b")
one_hot=one_hot_vectorizer.fit_transform(corpus).toarray()
#print(one_hot_vectorizer)
print(one_hot_vectorizer.get_feature_names())
print(one_hot)
sns.heatmap(one_hot,annot=True,cbar=False,xticklabels=['aravind', 'raamkumar', 's', 'sesagiri'],yticklabels=['S1','S2'])
