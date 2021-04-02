# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:56:41 2020

@author: postd
"""
import pandas as pd
import glob
import openpyxl as pyxl
import numpy as np
import os

def stateSequence(file_path,idp_file,fileDates,start,end):
    x = pyxl.load_workbook(idp_file)
    act = x.active
    data = act.values
    cols = next(data)
    data = list(data)
    idps = pd.DataFrame(data, columns=cols)
    
    idp_obs = idps.loc[:,["Families","Governorate"]]
    idp_obs = idp_obs.replace({"Salah al-Din":"Salahal Din",
                           "Thi-Qar":"Thi Qar"})
    obsDF = idp_obs.groupby("Governorate").agg("sum").reset_index()

    death_files = glob.glob(file_path + "Deaths_Data/*.csv")
    death_df = []
    for f in death_files:
        death_df.append(pd.read_csv(f))
    death_period = []
    for df in death_df:
        df.loc[:,"Week starting"] =\
            pd.to_datetime(df.loc[:,"Week starting"])
        death_period.append(df.loc[(df.loc[:,"Week starting"]>=start)&\
                               (df.loc[:,"Week starting"]<=end),:])
    deathDF = pd.concat(death_period,axis=0)
    deathDF = deathDF.fillna(0)
    deaths_05_06 = deathDF.sum(axis=0).astype(int)
    deaths_05_06 = deaths_05_06.sort_index()
    obsDF = obsDF.sort_values(by="Governorate")
    obsDF.loc[:,"deaths"] = deaths_05_06.values

    distances = pd.read_csv(file_path+"locations_dists.csv")
    distances.index = distances.locations
    distances = distances.drop(columns = "locations")

    obsDF = obsDF.merge(distances,left_on="Governorate",
                        right_index=True)

    pop = pd.DataFrame({"Gov":["Basrah","Muthanna","Qadissiya","Najaf",
                               "Erbil","Sulaymaniyah","Babylon","Baghdad",
                               "Dahuk","Thi Qar","Diyala","Anbar",
                               "Kerbala","Kirkuk","Missan","Ninewa",
                               "Salahal Din","Wassit"],
                    "Pop":[2818800,788300,1250200,1425700,1797700,
                           1990300,1999000,7877900,1252300,2029300,
                           1584900,1485900,1180500,1548200,1078100,
                           3612300,1544100,1335200]})
    obsDF = obsDF.merge(pop,left_on="Governorate",
                    right_on="Gov").drop(columns="Gov")

    weatherDF = pd.read_csv(file_path+"weather_iraq_govs.csv")
    weatherDF = weatherDF.iloc[:,1:]
    weatherDF.Date = pd.to_datetime(weatherDF.Date)
    weather = weatherDF.loc[weatherDF.Date==end,:]
    weather = weather.replace({"Wasit":"Wassit"})
    obsDF = obsDF.merge(weather.iloc[:,1:],left_on="Governorate",
                    right_on="Gov").drop(columns="Gov")

    obsDF.index = obsDF.Governorate
    obsDF = obsDF.drop(columns="Governorate")
    return obsDF

def main():
    path = "C://Users/postd/FoMiSimulator/"
    idp_files = glob.glob(path+"IDP_Lists/*.xlsx")
    idpFileDates = [pd.to_datetime(x[-17:-5]) for x in idp_files]
    idp_fs = pd.DataFrame({"Date":idpFileDates,"Files":idp_files})
    idp_fs = idp_fs.sort_values(by="Date").reset_index(drop=True)
    idpFileDates.sort()
    
    stateDFs = []
    for i in range(1,len(idpFileDates)):
        stateDFs.append(stateSequence(path,idp_fs.Files[i],idpFileDates,
                           idpFileDates[i-1],idpFileDates[i]))
        
    for df,date in zip(stateDFs,idpFileDates[1:]):
        df.to_csv(path+"StatesForSim/state_"+str(date.date())+".csv")
        
    states_by_location = []
    for locat in stateDFs[0].index:
        sbyL = []
        for df in stateDFs:
            sbyL.append(df.loc[locat,:])
        data = pd.concat(sbyL,axis=1)
        data.columns = [str(x.date()) for x in idpFileDates[1:]]
        data.to_csv(path+"StateDems/state_"+locat+".csv")

if __name__=="__main__":
    main()