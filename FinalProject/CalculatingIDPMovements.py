# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 14:28:09 2020

@author: postd
"""
import pandas as pd
import glob
import numpy as np

def prepActions(filename1,filename2):
    idps_0515 = pd.read_excel(filename1,sheet_name=0)

    idps_0615 = pd.read_excel(filename2,sheet_name=0)

    columns = [2,3,6,7] + list(range(8,28)) + list(range(36,66))

    idps_0515 = idps_0515.iloc[:,columns]
    idps_0515 = idps_0515.iloc[:,list(range(0,25))+list(range(38,54))]
    idps_0515 = idps_0515.iloc[:,list(range(0,24))+[25,26,27,29,
                                                    31,33,35,37,40]]

    idps_0615 = idps_0615.iloc[:,columns]
    idps_0615 = idps_0615.iloc[:,list(range(0,25))+list(range(38,54))]
    idps_0615 = idps_0615.iloc[:,list(range(0,24))+[25,26,27,29,
                                                    31,33,35,37,40]]

    idps_0515 = idps_0515.drop(columns=["Latitude","Longitude","Individuals"])
    idps_0615 = idps_0615.drop(columns=["Latitude","Longitude","Individuals"])
    
    counts_0515 = idps_0515.groupby(["Governorate","District"])\
        .agg("sum").reset_index()
        
    counts_0615 = idps_0615.groupby(["Governorate","District"])\
        .agg("sum").reset_index()
    
    change_05_06 = counts_0615.iloc[:,2:] - counts_0515.iloc[:,2:]
    change_05_06.loc[:,"Governorate"] = counts_0515.Governorate
    change_05_06.loc[:,"District"] = counts_0515.District 
    govDict = {"Thi-Qar":"Thi Qar","Salah al-Din":"Salahal Din"}
    change_05_06.Governorate = change_05_06.loc[:,"Governorate"]\
                                    .replace(govDict)
    
    change_gov = change_05_06.groupby("Governorate").agg("sum")\
        .reset_index()
        
    change_gov = change_gov.iloc[:,:20]
    actions_list = []
    for i in range(len(change_gov)):
        actions_list.append(change_gov.iloc[i,2:])
    return actions_list

def main():
    path = "C://Users/postd/FoMiSimulator/"
    idp_files = glob.glob(path+"IDP_Lists/*.xlsx")
    idpFileDates = [pd.to_datetime(x[-17:-5]) for x in idp_files]
    idp_fs = pd.DataFrame({"Date":idpFileDates,"Files":idp_files})
    idp_fs = idp_fs.sort_values(by="Date")
    
    actionDems = []
    for i in range(1,len(idp_fs)):
        actionDems.append(prepActions(idp_fs.Files[i-1],idp_fs.Files[i]))
    idpFileDates.sort()
    
    actions_by_loc = []
    for i in range(len(actionDems[0])):
        abyL = []
        for dems in actionDems:
            abyL.append(dems[i])
        df = pd.concat(abyL,axis=1)
        df.columns = [str(x.date()) for x in idpFileDates[1:]]
        df=df.astype(int)
        actions_by_loc.append(df)
    
    for x,y in zip(actions_by_loc,list(actions_by_loc[0].index)):
        x.to_csv(path+"ActionDems/Actions_"+y+".csv")
    
if __name__=="__main__":
    main()