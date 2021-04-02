# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:28:30 2020

@author: postd
"""

import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Embedding,LSTM,SimpleRNN
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pickle
import argparse
import tensorflow as tf
from glob import glob
import pandas as pd
tf.python.control_flow_ops = tf

def gen(data, label, batch_size):
    start = 0
    end = start + batch_size
    n = len(data)
    while True:
        observations  = data[start:end]
        
        obs_list = np.array(observations, dtype='float32')
        X_batch = obs_list
        
        y_batch = np.array(label[start:end], dtype='float32')
        
        start += batch_size
        end += batch_size
        if start >= n:
            start = 0
            end = batch_size
        
        yield (X_batch, y_batch)

def get_model(n_neurons,batchSize):
    
    opt = keras.optimizers.Adam()
    
    model = Sequential()
    # LSTM Implementation
    #model.add(LSTM(n_neurons,input_shape=(1,23),
                   #return_sequences=False,name="LSTM_Layer"))
    model.add(Dense(200,activation="relu",name="Dense_Layer1",
              input_shape=(23,)))
    model.add(Dense(100,activation="relu",name="Dense_Layer2"))
    model.add(Dense(50,activation="relu",name="Dense_Layer3"))
    model.add(Dense(18,name="Output_Layer"))
    model.compile(optimizer=opt, loss='mae')
    return model

def train_model(label,BATCH_SIZE,NUM_EPOCHS,lstm_neurons):
    data = pd.read_csv("StateDems/state_"+label+".csv",index_col=0)
    data = np.array([data.iloc[:,i].values\
                        for i in range(len(data.columns))])
    var_sequences = []
    for i in range(len(data)):
        var_sequences.append(data[i,:]) # FOR LSTM append([data[i,:]])
    var_seqs=np.array(var_sequences)
    labels = pd.read_csv("ActionDems/Actions_"+label+".csv",index_col=0)
    labels = np.array([labels.iloc[:,i].values\
                       for i in range(len(labels.columns))])
    lab_sequences = []
    for i in range(len(labels)):
        lab_sequences.append(labels[i,:])
    lab_seqs=np.array(lab_sequences)
    
    X_train, X_val, y_train, y_val = train_test_split(var_seqs,
                                                      lab_seqs,
                                                      test_size=0.2)
    # Get model
    model = get_model(lstm_neurons,BATCH_SIZE)

    model.summary()
    
    # Instantiate generators
    train_gen = gen(X_train, y_train, BATCH_SIZE)
    val_gen = gen(X_val, y_val, BATCH_SIZE)
    
    train_start_time = time.time()
    
    # Train model
    h = model.fit(x=train_gen,batch_size=BATCH_SIZE,
                  steps_per_epoch=X_train.shape[0],
                  epochs=NUM_EPOCHS,validation_steps=X_val.shape[0],
                  validation_data=val_gen)
    
    history = h.history
    
    total_time = time.time() - train_start_time
    print('Total training time: %.2f sec (%.2f min)' % (total_time,
                                                     total_time/60))
    
    # Save model architecture to model.json, model weights to model.h5
    json_string = model.to_json()
    with open('Models/model_'+label+'.json', 'w') as f:
        f.write(json_string)
    model.save_weights('Model_Weights/model_'+label+'.h5')
    
    # Save training history
    with open('Model_Hist/train_hist_'+label+'.p', 'wb') as f:
        pickle.dump(history, f)
        
    print('Model saved in model.json/h5, history saved in train_hist.p')

def main(args):
    num_epochs = int(args.num_epochs)
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    print('Training models')
    agent_locs = pd.read_csv("locations_defs.csv").locations.unique()
    for loc in agent_locs:
        train_model(loc,batch_size,num_epochs,300)
        print('DONE: Training model for '+loc)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs",default=1000)
    parser.add_argument("--batch_size",default=12)
    parser.add_argument("--learning_rate",default=0.01)
    args = parser.parse_args()
    main(args)