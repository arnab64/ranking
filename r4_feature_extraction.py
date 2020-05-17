#!/usr/bin/python
import numpy as np
import xgboost as xgb
import pandas as pd

# takes a query set and creates the search dataset. Later on the data-pipeline will be designed with this schema

static_list_queries = ["Post malone","charlie puth","last week tonight","trevor noah","hailee steinfeld","boeing","formula 1","lewis hamilton"]

# modules to be implemented:
# 1. query word match
# 2. query similarity method 1

ytdata = pd.read_csv("datasets/youtube_data.csv",error_bad_lines=False)
print("shape of ytdata", ytdata.shape)
for col in ytdata.columns:
        print(col)
#select a list of rows and cols
ytdata_sliced = ytdata.iloc[[i for i in range(20)],[0,1,2,3,4,5,6,]]
print(ytdata_sliced.shape)

def word_match(query,title):
    print(word1,word2,"matched=",set(query) & set(title))

