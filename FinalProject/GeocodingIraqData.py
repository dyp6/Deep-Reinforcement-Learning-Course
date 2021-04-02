# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:41:16 2020

@author: postd
"""
import pandas as pd
import glob
import geopandas as gpd
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import matplotlib.pyplot as plt
import tqdm
from functools import partial
import re

def gecodeWeatherData(path):
    weather = pd.read_csv(path+"iraq_weather.csv")
    weather.Date = pd.to_datetime(weather.Date,format="%Y%m%d",errors="coerce")

    weather = weather.loc[(weather.Date>=pd.to_datetime("2015-05-21"))&\
                          (weather.Date<=pd.to_datetime("2016-12-22")),:]
    weather = weather.loc[:,["Date","TEMP","PRCP","Lat","Long"]]
    weather.loc[:,"geom"] = weather.Lat.map(str) +\
                            "," + weather.Long.map(str)

    locator = Nominatim(user_agent="myGeocoder", timeout=1)
    reverse = partial(locator.reverse,language="en")
    rgeocode = RateLimiter(reverse, min_delay_seconds=0)

    tqdm.tqdm.pandas()

    weather.loc[:,"gov"] = weather.loc[:,"geom"].progress_apply(rgeocode)

    for i in weather.index:
        try:
            weather.loc[i,"gov_new"] = re.findall(r"[a-zA-Z]+\sGovernorate",
                                     weather.loc[i,"gov"][0])[0]\
                                    .replace(" Governorate","")
        except:
            try:
                weather.loc[i,"gov_new"] = re.findall(r"[a-zA-Z]+\sDistrict",
                                      weather.loc[i,"gov"][0])[0]\
                                        .replace(" District","")
            except:
                weather.loc[i,"gov_new"] = re.findall(r"[a-zA-Z]+\sProvince",
                                      weather.loc[i,"gov"][0])[0]\
                                        .replace(" Province","")

    weather = weather.drop(columns=["gov","Lat","Long","geom"])
    return weather

def cleanWeather(df)
    df = df.replace("Qar","Thi Qar")\
        .replace("Saladin","Salahal Din")\
        .replace("Sulaimaniya","Sulaymaniyah")\
        .replace("Halabja","Sulaymaniyah")\
        .replace("Qadisiyah","Qadissiya")\
        .replace("Maysan","Missan")\
        .replace("Dohuk","Dahuk")\
        .replace("Basra","Basrah")\
        .replace("Babil","Babylon")\
        .replace("Karbala","Kerbala")\
        .replace("Wassit","Wassit")
    
    weather_gov=df.groupby(["gov_new","Date"]).agg("max").reset_index()
    nin = weather_gov.loc[weather_gov.gov_new.isin(["Erbil","Dahuk"]),:]\
    .groupby("Date").agg("mean").reset_index().loc[:,["Date","TEMP","PRCP"]]
    nin.loc[:,"gov_new"] = "Ninewa"
    weather_gov = pd.concat([weather_gov,nin])
    weather_gov = weather_gov.loc[weather_gov.gov_new!="Khuzestan",:]
    idp_files = glob.glob(path+"IDP_Lists/*.xlsx")
    idpFileDates = [pd.to_datetime(x[-17:-5]) for x in idp_files]
    idpFileDates.sort()

    dates = []
    temps = []
    prcps = []
    govs = []
    for gov in weather_gov.gov_new.unique():
        for i in range(1,len(idpFileDates)):
            dates.append(idpFileDates[i])
            govs.append(gov)
            temps.append(round(weather_gov.loc[(weather_gov.gov_new==gov)&\
                                (weather_gov.Date>=idpFileDates[i-1])&\
                                (weather_gov.Date<=idpFileDates[i]),
                                "TEMP"].values.mean(),2))

            prcps.append(round(weather_gov.loc[(weather_gov.gov_new==gov)&\
                                (weather_gov.Date>=idpFileDates[i-1])&\
                                (weather_gov.Date<=idpFileDates[i]),
                                "PRCP"].values.mean(),2))
            
    final_weather = pd.DataFrame({"Date":dates,"Gov":govs,
                              "Temp":temps,"Precip":prcps})
    final_weather = final_weather.fillna(method="backfill")
    return final_weather

def main():
    file_path = "C://Users/postd/Documents/Research_Stuff/"
    weatherDF = geocodeWeatherData(file_path)
    finWeather = cleanWeather(weatherDF)
    finWeather.to_csv(file_path+"weather_iraq_govs.csv")
    
if __name__=="__main__":
    main()