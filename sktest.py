# -*- coding: utf-8 -*-
"""
Created on Tue May 18 22:10:35 2021

@author: Niu
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from data_process import *
import numpy as np


x , y = load_iris(return_X_y=True)
#print(x,y)


train_set = Open_data('Train.csv')[:]
d , l = train_set[0] , train_set[1]
d = np.array(d , dtype=np.float32)
l = np.array(l)
print(d , l)
total_dict = {}
for i in l:
 if i not in total_dict:
   total_dict[i] = 0
 else:
   total_dict[i] += 1
print(total_dict.items())   

clf = LogisticRegression(random_state=0).fit(d,l)
#print(clf.decision_function(d[1]))
print(clf.score(d,l))