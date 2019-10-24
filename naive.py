#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 21:50:40 2019

@author: amogh
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

dataset=pd.read_csv("/home/amogh/BE-4247/LP1 Shubham/asdfghjl/Assignments/DA 2/Pima.csv");

dataset.describe()
dataset.info()
dataset.dtypes

plt.hist(dataset.iloc[:,1],bins=25)
plt.xlabel("IDK");
plt.ylabel("Frequency");
plt.title("IDK vs Frequency");
plt.show()


dataset.isnull().any()
dataset=dataset.fillna(method='ffill')


from sklearn.model_selection import train_test_split

X=dataset.iloc[:,0:7]
Y=dataset.iloc[:,8]

from sklearn.preprocessing import StandardScaler
scx=StandardScaler()
X=scx.fit_transform(X)

xtrain, xtest, ytrain, ytest= train_test_split(X,Y,test_size=0.3,random_state=0)


from sklearn.naive_bayes import GaussianNB

classifier= GaussianNB()
classifier.fit(xtrain,ytrain)
ypred=classifier.predict(xtest)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ypred,ytest)

dataset2=dataset.iloc[:,0:4]


dataset2.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

sns.heatmap(cm, annot=True)
