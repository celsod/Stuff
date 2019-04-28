#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:09:10 2019

@author: celso
using code from Arnaud De Launay and his twitter sentiment challenge program
"""
import tweepy
from textblob import TextBlob
from textblob.taggers import PatternTagger
from textblob.sentiments import PatternAnalyzer
import numpy as np
import operator

#the consumer key, consumer secret, access token, and access token secret were all generated from twitter.  The developer option was
#added to my bullshit twitter account

consumer_key = 'qEUem1dL8smMEZ27CHouI6fNS'
consumer_secret = 'JQtncM80jhedhZ8cLS7sNEyftnkQQaWMjPn6cztgtosyiIkgFB'

access_token = '1122520496980078593-OuhZQPawHhJkmoMQjM94zatBUkb70p'
access_token_secret = 'ckYsmX1bQKptcpcwxu9bcwvTMqLvrcxQwUipe7B59tLnS'

authentication = tweepy.OAuthHandler(consumer_key, consumer_secret) #gaining access to twitter via tweepy first half of authentication
authentication.set_access_token(access_token, access_token_secret) #second half of authentication

#main variable that will start gathering the twitter information
api = tweepy.API(authentication)

#creating an array with the possible twitter callouts for Euro-Dollar forex pair
forex_pair = ['eurusd', 'EURUSD', '$EUR$USD', '$EURUSD', 'usdeur', 'USDEUR', '$USD$EUR', '$USDEUR']

#creating a function to label the sentiment of the tweets that are captured
#source is from Arnaud De Launay
def get_label (analysis, threshold = 0):
    if analysis.sentiment[0]>threshold: #only need to look at the current tweet.No need to look at the previously evaluated tweets
        return 'Positive'
    else:
        return 'Negative'
    
#this is where we will start to retrieve tweets and begin analyzing them
#source is from Arnaud De Launay
all_polarities = dict() #putting all polarities into a dictionary variable since these will be words either positive or negative

for pair in forex_pair: #starting a for loop since there are multiple ways to describe the Euro-Dollar forex pair
    
    this_pair_polarities = [] #initializing the dictionary with no values
    
    this_pair_tweets = api.search(pair) #searching in twitter for tweets with the forex pair name
    
    with open ('%s_tweets.csv' % pair, 'w') as this_pair_file: #opening a .csv file
        
        this_pair_file.write('tweet,sentiment_label\n') #writing to the .csv file
        
        for tweet in this_pair_tweets: #for loop to go through the tweets
            
            analysis = TextBlob(tweet.text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()) #putting the tweets into a variable called analysis
            
            this_pair_polarities.append(analysis.sentiment[0]) #adding to the dictionary the sentiment analysis of the tweet 
            
            this_pair_file.write('%s,%s\n' % (tweet.text.encode('utf8'), get_label(analysis))) #writing to .csv file the tweet and the
                                                                                                # label from the get_label function from the analysis of the tweet
    all_polarities[pair] = np.mean(this_pair_polarities) #putting the average of the polarity for the forex pair descriptor into the dictionary

sorted_analysis = sorted(all_polarities.items(), key=operator.itemgetter(1), reverse=True) #sorting the polarities of all items
print ('Mean Sentiment Polarity in descending order :') #output of program
for pair, polarity in sorted_analysis: #for loop to go through the dictionary of items
    print ('%s : %0.3f' % (pair, polarity)) #print out of the dictionary