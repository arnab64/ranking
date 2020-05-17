#!/usr/bin/python
import numpy as np
import xgboost as xgb
import pandas as pd

# read data and change it to format needed by data sciences
print("Reading data!\n")
#ytdata = pd.read_csv("/mnt/c/Users/arnab/Documents/MediaInitiative/dataset/USvideos.csv", error_bad_lines=False)

ytdata = pd.read_csv("../datasets/youtube_data.csv",error_bad_lines=False)
print("shape of ytdata", ytdata.shape)
for col in ytdata.columns:
    print(col)
#ytdata.to_csv('youtube_data.csv')

ytdatanumeric = pd.read_csv("../datasets/youtube_data_numeric.csv",error_bad_lines=False)
print("Shape of Youtube numeric data", ytdatanumeric.shape)
for colx in ytdatanumeric.columns:
    print(colx)

# getting the video titles in a text file for training fasttext model
print(ytdata.title.head(100))
np.savetxt('../datasets/titles_np.txt', ytdata.title, fmt='%s',encoding='utf8')
