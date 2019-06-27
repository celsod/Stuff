#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:56:58 2019

@author: celso

Sentdex cryptocurrency RNN example
"""

import pandas as pd
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

SEQ_LEN = 240 # Look back 240 minutes (4 hours) for the neural network; can change as necessary
FUTURE_PERIOD_PREDICT = 10 # Predict 10 minutes into the future
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PREDICT-{int(time.time())}"


def classify(current, future): # Beginning of model determining what is a positive output
    if float(future) > float(current):
        return 1
    else:
        return 0
# This function states that if the future is greater than the current, then it's a positive output
# Need to consider the sell side of the logic



def preprocess_df(df):
    df = df.drop('FUTURE', 1) # no need to preprocess the FUTURE column 
    
    for col in df.columns: # iterate over all of the columns 
        if col != 'TARGET': # normalize the data except the TARGET column 
            df[col] = df[col].pct_change() # the .pct_change() normalizes the change in price data 
            df.dropna(inplace=True) # remove any NA values that were created by the normalization process
            df[col] = preprocessing.scale(df[col].values) # scale the normalized values between 0 and 1
            
    df.dropna(inplace=True) # remove any newly created NA values
    
    sequential_data = [] # empty list that will contain the sequences
    prev_days = deque(maxlen=SEQ_LEN) # will keep the list up to date and remove old entries so that the 
    # length of the prev_days list is always filled with the most up to date information and only 4 hours
    # worth of data
    
    for i in df.values: # iterate over the values in the dataframe
        prev_days.append([n for n in i[:-1]]) # store all values except the TARGET value
        if len(prev_days) == SEQ_LEN: # ensure there are 240 minutes worth of data
            sequential_data.append([np.array(prev_days), i[-1]]) # append the data to ensure that there are
            # no TARGET values in the data
    random.shuffle(sequential_data) # shuffle the data for good measure.
    
    buys = []
    sells = []
    
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    
    random.shuffle(buys)
    random.shuffle(sells)
    
    lower = min(len(buys), len(sells))
    
    buys = buys[:lower] # up to the lower value
    sells = sells[:lower]
    
    sequential_data = buys+sells
    random.shuffle(sequential_data)
    
    X = []
    Y = []
    
    for seq, target in sequential_data:
        X.append(seq)
        Y.append(target)
    
    return np.array(X), Y


df = pd.DataFrame() # initialize an empty Dataframe

df = pd.read_csv('ADD YOUR FILE PATH HERE.csv',
                 names=["Date Time", "OPEN", "HIGH", "LOW", "CLOSE"],
                 parse_dates=True,
                 infer_datetime_format=True,
                 index_col=[0])

print(df.head())

df.fillna(method='ffill', inplace=True)
df.dropna(inplace=True)

df['FUTURE'] = df['CLOSE'].shift(-FUTURE_PERIOD_PREDICT) # not a real prediction, just shifting the close
# by the future period predict value
df['TARGET'] = list(map(classify, df["CLOSE"], df["FUTURE"]))

df.dropna(inplace=True)

print(df.head())

times = sorted(df.index.values) # pulling the times from the dataframe
#print(times[0])
last_5pct = sorted(df.index.values)[-int(0.05*len(times))] # establishing the last 5% of the previous times
#print(last_5pct)

validation_df = df[(df.index >= last_5pct)] # creating a validation set with the last 5% of times
#print(validation_df.head())
training_df = df[(df.index < last_5pct)] # the original dataframe has everything else up to the last 5%
#print(training_df.head())

train_x, train_y = preprocess_df(df)
validation_x, validation_y = preprocess_df(df)

print(f'Train data: {len(train_x)} validation: {len(validation_x)}') 
print(f'Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}')
print(f'VALIDATION DONT BUYS: {validation_y.count(0)}, VALIDATION BUYS: {validation_y.count(1)}')

model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6) # optimizer call out for the model build

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

filepath = "RNN_final-{epoch:02d}-{val_acc:.3f}"
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, verbose=1, save_best_only=True, mode='max'))

history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, 
                    validation_data=(validation_x, validation_y),
                    callbacks=[tensorboard, checkpoint])














