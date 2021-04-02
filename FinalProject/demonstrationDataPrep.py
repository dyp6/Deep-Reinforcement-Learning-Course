# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:28:05 2020

@author: postd
"""

# Need to get the IOM DTM Data for iraq in the correct form
# For each date, need the number of IDPs at each location and
# the proportion of the population moving from each location
# to every other location.

import pandas as pd
import numpy as np

def prepDemonstrations(IDP_mvmnts):
    gov_Movs = IDP_mvmnts.groupby(["Date","Origin","Dest"])\
        .agg("sum").reset_index()

    gov_Movs = gov_Movs.sort_values(by=["Date","Origin","Dest"])\
        .reset_index(drop=True)
    return gov_Movs

# Want to add year/date capabilities to this so it can handle more 
# than just the data I currently am using for this project
def action_obs_sequences(movement_df):
    movement_df.Date = pd.to_datetime(movement_df.Date)
    movement_df = movement_df.sort_values(by="Date")
    # Extract action sequence
    a_list = []
    for loc in movement_df.Origin.unique():
        L1_a = movement_df.loc[movement_df.Origin==loc,['Date','Dest',
                                                        'fled']]

        temp_a_list = []
        d_list = []
        for date in L1_a.Date.unique():
            d_list.append(date)
            x = L1_a.loc[L1_a.Date==date,['Dest','fled']]
            x.index = x.Dest
            x = x.drop(columns="Dest")
            x = x.T
            x.index = [loc+"_"+str(pd.to_datetime(date).date())]
            temp_a_list.append(x)
        a_list.append(temp_a_list)
    
    action_DFs = []
    for i in range(len(a_list)):
        df = pd.concat(a_list[i],axis=0)
        df.index = d_list
        action_DFs.append(df)

    for df,name in zip(action_DFs,movement_df.Origin.unique()):
        df.to_csv("Demonstration_Actions/actionSequence_"+name+".csv")
    
    movement_df.to_csv("Grouped_IDP_counts_15-17.csv")
    # Put together observation sequence data
    tot_moves = movement_df.groupby(["Date","Dest"]).agg("sum")
    tot_moves = tot_moves.reset_index()

    tot_leaves = movement_df.groupby(["Date","Origin"]).agg("sum")
    tot_leaves = tot_leaves.reset_index()


    tot_moves.loc[tot_moves.Dest.isin(tot_leaves.Origin),"fled"] =\
        tot_moves.loc[tot_moves.Dest.isin(tot_leaves.Origin),"fled"].values-\
            tot_leaves.fled.values
    

    dists = pd.read_csv("locations_dists.csv")
    state_info = pd.read_csv("Governorate_state_info.csv")
    state_info= state_info.loc[:,["Date","gov","death","temp",
                              "precip","Gov_eventsAll"]]
    state_info.Date = pd.to_datetime(state_info.Date)
    state_info = state_info.sort_values(by="Date")
    obs_list = []
    tot_list = []
    for date in movement_df.Date.unique():
        obs_list.append(state_info.loc[state_info.Date == date,:])
        tot_list.append(tot_moves.loc[tot_moves.Date==date,:])
    
    start_pops = pd.read_csv("start_idp_pops.csv")

    idp_pops = [start_pops.tot_idps.values]
    for df in tot_list:
        new_idps = df.fled.values + idp_pops[-1]
        idp_pops.append(new_idps)
    
    combined = []
    for idps,obs in zip(idp_pops[:-1],obs_list):
        merged = obs.merge(dists,left_on="gov",right_on="locations")
        merged.index = merged.gov
        merged.loc[:,"tot_idps"] = idps
        merged = merged.drop(columns = ["Date","gov","locations"])
        combined.append(merged)
    
    for df,d in zip(combined,d_list):
        df.to_csv("Demonstration_Observations/"+\
              str(pd.to_datetime(d).date())+".csv")

    for df,d in zip(combined,d_list):
        env_df = df.drop(columns=["tot_idps"])
        env_df.to_csv("External_Environment_Data/"+\
                      str(pd.to_datetime(d).date())+".csv")
            
def main():
    IDP_movements = pd.read_csv("C://Users/postd/Documents/\
FoMiSimulator/IDP_counts_15-17.csv")

    moves = prepDemonstrations(IDP_movements)
    action_obs_sequences(moves)
    
    
if __name__=="__main__":
    main()