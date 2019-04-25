#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 19:58:31 2019

@author: celso
"""

#Trying the same intro program but using a Stochastic Gradient Descent classifier
from sklearn.linear_model import SGDClassifier

#data copied and pasted from the intro program
X = [[181,80,44], [177,70,43], [160,60,38], [154,54,37], 
     [166,65,40], [190,90,47], [175,64,39], [177,70,40],
     [168,70,38], [210,90,50], [164,28,30]]

Y = ['male', 'female', 'male', 'female', 'female', 'male',
     'female', 'male', 'male', 'female', 'female']

#setting the variable clf to a SGD classifier; will need to pass inputs to the algorithm

clf = SGDClassifier(shuffle = True, alpha = 0.01) #setting shuffle to True implies that the algorithm will randomize the input data
#setting alpha 0.01 is the regularization term that is used to compute the learning rate; smaller number means more steps

#fitting the variable clf to the data given the previously called algorithm
clf = clf.fit(X,Y)
#using the new fitted variable to predict the input data
prediction = clf.predict ([[168,64,49]])

print (prediction)