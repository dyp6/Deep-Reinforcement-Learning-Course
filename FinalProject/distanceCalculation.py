# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 13:28:10 2020

@author: postd
"""

import pandas as pd
import requests, json
    
def getTravelDists(filename):
    with open(".credentials/APIKeysAndOtherCredentials.txt",'r') as text:
        api_key = text.readlines()[0].split("=")[1]
    
    url='https://maps.googleapis.com/maps/api/distancematrix/json?'
    locations = pd.read_csv(filename)
    locations.loc[:,"latlongpairs"] = [str(locations.latitude.values[i])+","+\
        str(locations.longitude.values[i]) for i in range(len(locations))]
    dist_km=[]
    
    for i in range(len(locations)):
        origin = locations.latlongpairs.values[i]
        dest = "|".join([x for x in locations.latlongpairs])
        r = requests.get(url + 'origins=' + origin +
                     '&destinations=' + dest +
                     '&key=' + api_key)
        x = r.json()
        
        dists = []
        for g in range(len(locations)):
            dists.append(x["rows"][0]['elements'][g]['distance']['text'])
        dist_km.append(dists)
        
    locInfo = pd.DataFrame(columns=[x + "_distKm" for x in locations.locations],
                       index=locations.locations)
    for i in range(len(locInfo)):
        locInfo.iloc[i,:] = [float(x.strip(" km")\
                               .replace(",","")) for x in dist_km[i]]
    return locInfo

def main():
    dists = getTravelDists("C://Users/postd/Documents\
/FoMiSimulator/locations_defs.csv")
    dists = dists.replace(1.,0.)
    dists.to_csv("C://Users/postd/Documents/\
FoMiSimulator/locations_dists.csv")

if __name__ == "__main__":
    main()
