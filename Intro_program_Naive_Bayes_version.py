#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 19:19:29 2019

@author: celso
"""
#Trying the same intro problem but with a Naive Bayes estimator

from sklearn.naive_bayes import GaussianNB #callilng the GaussianNB function from the sklearn library

clf = GaussianNB() #Setting the clf variable to the GaussianNB algorithm

#data copied and pasted from the intro program
X = [[181,80,44], [177,70,43], [160,60,38], [154,54,37], 
     [166,65,40], [190,90,47], [175,64,39], [177,70,40],
     [168,70,38], [210,90,50], [164,28,30]]

Y = ['male', 'female', 'male', 'female', 'female', 'male',
     'female', 'male', 'male', 'female', 'female']

clf = clf.fit(X,Y) #fitting the data set using the predefined algorithm from earlier

prediction = clf.predict([[190,70, 45]]) #predicting output given the input after fitting the given data set

print (prediction)