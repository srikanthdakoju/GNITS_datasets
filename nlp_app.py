# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 14:07:24 2018

@author: srikanth
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.externals import joblib
classifier = joblib.load('review.pkl') 



text = 'i hate this hotel'

doc1 = re.sub('[^a-zA-Z]', ' ', text)
doc1 = doc1.lower()
doc1 = doc1.split()
ps = PorterStemmer()
doc1 = [ps.stem(word) for word in doc1 if not word in set(stopwords.words('english'))]
doc1= ' '.join(doc1)
doc2 = doc1.split()
print(doc2)


dicti =cv.get_feature_names()
check=[]

for i in range(len(dicti)):
    k=0
    for j in range(len(doc2)):
        if doc2[j]==dicti[i]:
            k=1
    if k==1:
        check.append(1)
        ## Mapping Features
    else:
        check.append(0)
check1 = np.asarray(check)
check1 = np.reshape(check1,(1,1565))

## Prediction Using Model
real_pred = classifier.predict(check1)
if real_pred[0]==0:
    Comments ="This review is negative"
if real_pred[0]==1:
    Comments = "This review is positive"