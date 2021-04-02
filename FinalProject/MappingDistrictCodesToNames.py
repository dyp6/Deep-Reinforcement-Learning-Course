# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 20:35:46 2020

@author: postd
"""

import pandas as pd
import numpy as np

migDyads = pd.read_csv("C://Users/postd/Downloads/migration_dyads_new.csv")

districts = [str(migDyads.calc_district[i])+"_"+\
             str(migDyads.calc_district_code[i]) \
                 for i in range(len(migDyads))]
    
districts = [x for x in pd.Series(districts).unique()]
dist_model_name = [x.split("_")[0] for x in districts]
dist_model_code = [x.split("_")[1] for x in districts]

ocha = pd.read_csv("C://Users/postd/Documents/Research_Stuff\
/OCHA_dist_names.csv")

ocha = [x for x in ocha.Districts.values]

[x for x in ocha if x not in dist_model_name]
[x for x in dist_model_name if x not in ocha]
