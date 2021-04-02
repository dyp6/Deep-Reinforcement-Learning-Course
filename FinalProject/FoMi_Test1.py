# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:36:45 2020

@author: postd
"""

from FoMiEnv import FoMiEnv
import pandas as pd
import numpy as np
from keras.models import model_from_json
from keras.optimizers import Adam
from glob import glob
import matplotlib.pyplot as plt
import argparse
import warnings

def loadTrainedNetworks(locations):
    network_list = {}
    for loc in locations:
        opt = Adam()
        model_file = "Models/model_"+loc+".json"
        with open(model_file, 'r') as jfile:
            model = model_from_json(jfile.readline())

        model.compile(opt, "mse")
        weights_file = "Model_Weights/model_"+loc+".h5"
        model.load_weights(weights_file)
        network_list[loc] = model
    return network_list

def runOnTrainingData(environment,nets_dict,
                      locations,dates):
    # TRAINING ERROR
    actions_all_train = []
    state_train = environment.reset().copy()
    states_train = [state_train]
    for i in range(len(dates[:32])):
        actions_step_train = []
        for j in range(len(locations)):
            action_train = nets_dict[locations[j]]\
                .predict(state_train.loc[locations[j],:]\
                         .to_numpy()[None,:]) # LSTM [None,None,:]
            new_state_train = environment\
                .step(action_train[0],locations[j])
            actions_step_train.append(action_train[0])
        state_train = new_state_train.copy()
        actions_all_train.append(actions_step_train)
        states_train.append(state_train)
    return states_train, actions_all_train

def runOnTestingData(environment,nets_dict,
                      locations,dates,starting_state):
    # TESTING ERROR
    actions_all_test = []
    state_test = starting_state.copy()
    states_test = [state_test]
    for i in range(len(dates[32:])):
        actions_step_test = []
        for j in range(len(locations)):
            action_test = nets_dict[locations[j]]\
                .predict(state_test.loc[locations[j],:]\
                         .to_numpy()[None,:]) # LSTM [None,None,:]
            new_state_test = environment\
                .step(action_test[0],locations[j])
            actions_step_test.append(action_test)
        state_test = new_state_test.copy()
        actions_all_test.append(actions_step_test)
        states_test.append(state_test)
    return states_test, actions_all_test

def evalSimResults(states_train,states_test,dates):
    val_files = glob("StatesForSim/*.csv")
    train_val_files = val_files[:33]
    test_val_files = val_files[33:]

    train_val_dfs = []
    test_val_dfs = []
    for train in train_val_files:
        train_val_dfs.append(pd.read_csv(train,index_col=0))
    
    for test in test_val_files:
        test_val_dfs.append(pd.read_csv(test,index_col=0))

    idpPops_train = [states_train[i].iloc[:,0] for\
                     i in range(len(states_train))]
    idpPops_test = [states_test[i].iloc[:,0] for \
                    i in range(1,len(states_test)-1)]

    train_errors = [(train_val_dfs[i].Families.values - idpPops_train[i]) \
                    for i in range(len(idpPops_train))]
    test_errors = [(test_val_dfs[i].Families.values - idpPops_test[i]) \
                    for i in range(len(idpPops_test))]
    
    modelTrainDF = pd.DataFrame(columns=train_val_dfs[0].index)
    for i in range(len(modelTrainDF.columns)):
        modelTrainDF.iloc[:,i] = \
            [idpPops_train[j][i] for j in range(len(idpPops_train))]
    modelTrainDF.loc[:,"Date"] = dates[:33]

    modelTestDF = pd.DataFrame(columns=test_val_dfs[0].index)
    for i in range(len(modelTestDF.columns)):
        modelTestDF.iloc[:,i] = \
            [idpPops_test[j][i] for j in range(len(idpPops_test))]
    modelTestDF.loc[:,"Date"] = dates[33:]

    valTrainDF = pd.DataFrame(columns=train_val_dfs[0].index)
    for col in list(valTrainDF.columns):
        idps = []
        for j in range(len(train_val_dfs)):
            idps.append(train_val_dfs[j]\
                    .loc[col,'Families'])
        valTrainDF.loc[:,col] = idps
    valTrainDF.loc[:,"Date"] = dates[:33]

    valTestDF = pd.DataFrame(columns=test_val_dfs[0].index)
    for col in list(valTestDF.columns):
        idps = []
        for j in range(len(test_val_dfs)):
            idps.append(test_val_dfs[j]\
                    .loc[col,'Families'])
        valTestDF.loc[:,col] = idps
    valTestDF.loc[:,"Date"] = dates[33:]
    return modelTrainDF,valTrainDF,modelTestDF,valTestDF

def plotResults(mTrDF,vTrDF,mTeDF,vTeDF,model_label):
    for locat in list(mTrDF.columns[:-1]):
        fig, ax = plt.subplots(1,figsize=(8,4),sharex=True)

        ax.plot(mTrDF.Date,mTrDF.loc[:,locat],
            color="blue",linewidth=2,
            label="Model Simulated IDP Counts (Training)")
        ax.plot(vTrDF.Date,vTrDF.loc[:,locat],color="red",linewidth=2,
                label="Actual IDP Counts (Training)")
        ax.set_ylabel("Number of IDPs")
        ax.legend()
        fig.suptitle(locat+" Governorate (Training) Model Results vs. Actual Counts")
        fig.savefig("TrainResults/"+locat+"_"+model_label+"_TrainResults.png")
        plt.close()

    for locat in list(mTeDF.columns[:-1]):
        fig, ax = plt.subplots(1,figsize=(8,4),sharex=True)

        ax.plot(mTeDF.Date,mTeDF.loc[:,locat],
            color="blue",linewidth=2,
            label="Model Simulated IDP Counts (Testing)")
        ax.plot(vTeDF.Date,vTeDF.loc[:,locat],color="red",linewidth=2,
                label="Actual IDP Counts (Testing)")
        ax.set_ylabel("Number of IDPs")
        ax.legend()
        fig.suptitle(locat+" Governorate (Testing) Model Results vs. Actual Counts")
        fig.savefig("TestResults/"+locat+"_"+model_label+"_TestResults.png")
        plt.close()

def main(args):
    path = "C://Users/postd/FoMiSimulator/"
    state_files = glob(path+"StatesForSim/state_*.csv")
    stateFileDates = [pd.to_datetime(x[-14:-4]) for x in state_files]
    state_fs = pd.DataFrame({"Date":stateFileDates,
                             "Files":state_files})
    state_fs = state_fs.sort_values(by="Date").reset_index(drop=True)
    stateFileDates.sort()
    locations = list(pd.read_csv(state_fs.Files[0],index_col=0).index)
    action_nets = loadTrainedNetworks(locations)

    env = FoMiEnv("StatesForSim/state_2015-06-04.csv","StatesForSim",
                  stateFileDates)
    
    train_states, train_actions = runOnTrainingData(env,action_nets,
                                                locations,stateFileDates)
    
    test_states, test_actions = runOnTestingData(env,action_nets,locations,
                                             stateFileDates,train_states[-1])

    modelTrDf,valTrDf,modelTeDf,valTeDf=evalSimResults(train_states,
                                                       test_states,
                                                       stateFileDates)

    plotResults(modelTrDf,valTrDf,modelTeDf,valTeDf,args.net_label)

if __name__=="__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("net_label",help="Give a label for the network\
                       architecture being used for this run.")
    args = parser.parse_args()
    main(args)