#!/usr/bin/python
import numpy as np
import xgboost as xgb
import pandas as pd

#split original dataset into numeric and non-numeric dataset
#non-numeric dataset will undergo further processin
#all datasets will be later joined into one cohesive dataset

#ytdata = pd.read_csv("/mnt/c/Users/arnab/Documents/MediaInitiative/dataset/USvideos.csv",error_bad_lines=False)
#ytdata = pd.read_csv("datasets/youtube_data_numeric.csv")
class getDmatrix:
    def __init__(self):
        self.qtdata = pd.read_csv("../datasets/query_title_data.csv")
        for col in self.qtdata.columns:
            print("label",col)

    def splitthedata(self):
        self.qtdata.sample(frac=0.8,random_state=0)
        print("found",len(self.qdata.columns),self.qdata.columns)        
        ytdata['views'] = ytdata['views'].astype(float)
        ytdata['likes'] = ytdata['likes'].astype(float)
        ytdata['dislikes'] = ytdata['dislikes'].astype(float)
        ytdata['comment_total'] = ytdata['comment_total'].astype(float)
        dtrain = xgb.DMatrix('datasets/ytdata_numericonly.csv?format=csv&label_column=0')
#ytdata = ytdata.loc[:, ~ytdata.columns.str.contains('^Unnamed')]
#del ytdata['date']
#del ytdata['category_id']

#ytdata.to_csv("datasets/ytdata_numericonly.csv", index=False)
dtrain = ytdata.sample(frac=0.75, random_state=0)
dtest = ytdata.drop(dtrain.index)
dtrain.to_csv("datasets/ytprocessed_train.csv", index=False)
dtest.to_csv("datasets/ytprocessed_test.csv", index=False)

labels = []
for col in ytdata.columns:
    print("label added",col)
    labels.append(col)

ncol = len(labels)
print("Found", ncol, "labels are these:", labels)

print("\n\ndatatypes\n",ytdata.dtypes)
print(ytdata.head(5))
#['category_id', 'views', 'likes', 'dislikes', 'comment_total']
ytdata['views'] = ytdata['views'].astype(float)
ytdata['likes'] = ytdata['likes'].astype(float)
ytdata['dislikes'] = ytdata['dislikes'].astype(float)
ytdata['comment_total'] = ytdata['comment_total'].astype(float)
print("\nnew datatypes\n",ytdata.dtypes)
#ytdata['views'] = ytdata.to_numeric(ytdata['views'], downcast='float')

#data = pandas.DataFrame(np.arange(12).reshape((4,3)), columns=['a', 'b', 'c'])
#label = pd(np.random.randint(2, size=4))
#print("label=", label)
#dtrain = xgb.DMatrix(data, label=label)
#this method is not directly working, hence 
#dtrain_1 = xgb.DMatrix(ytdata,label=labels)

# loading csv file directly into DMatrix
dtrain = xgb.DMatrix('datasets/ytdata_numericonly.csv?format=csv&label_column=0')
dtrain.save_binary("binaries/dtrain.buffer")
