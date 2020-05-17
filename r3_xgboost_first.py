#!/usr/bin/python
import numpy as np
import xgboost as xgb
import pandas as pd

#ytdata = pd.read_csv("/mnt/c/Users/arnab/Documents/MediaInitiative/dataset/USvideos.csv",error_bad_lines=False)
dtrain_s = pd.read_csv("datasets/ytprocessed_train.csv")
dtest_s = pd.read_csv("datasets/ytprocessed_test.csv")
#for col in ytdata.columns:
#    print(col)
print(dtrain_s.shape)
print(dtrain_s.head(10))
dtrain_formatted = xgb.DMatrix(dtrain_s, label=dtrain_s.views)
dtest_formatted = xgb.DMatrix(dtest_s, label=dtest_s.views)
#xgb.DMatrix(data =as.matrix(train.x), label= train.y)
#ytdata = xgb.load("binaries/dtrain.buffer")
# specify parameters via map, definition are same as c++ version
param_logreg = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
param_rankndcg = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'rank:ndcg'}
param_rankndcg_d10 = {'max_depth':10, 'eta':1, 'silent':1, 'objective':'rank:ndcg'}
param_rankndcg_d100 = {'max_depth':100, 'eta':1, 'silent':1, 'objective':'rank:ndcg'}
#dtrain = ytdata.sample(frac=0.75, random_state=0)
#dtest = ytdata.drop(dtrain.index)
#dtrain = xgb.DMatrix('datasets/ytprocessed_train.csv?format=csv&label_column=0')
#dtest = xgb.DMatrix('datasets/ytprocessed_test.csv?format=csv&label_column=0') 
print("shape of traindataset",dtrain_s.shape)
print("shape of testdataset:",dtest_s.shape)

# specify validations set to watch performance
watchlist = [(dtest_formatted, 'eval'), (dtrain_formatted, 'train')]
num_round = 20
rank_model_1 = xgb.train(param_rankndcg, dtrain_formatted, num_round, watchlist)
rank_model_2 = xgb.train(param_rankndcg_d10, dtrain_formatted, num_round, watchlist)
rank_model_3 = xgb.train(param_rankndcg_d100, dtrain_formatted, num_round, watchlist)
# this is prediction
preds_1 = rank_model_1.predict(dtest_formatted)
preds_2 = rank_model_2.predict(dtest_formatted)
preds_3 = rank_model_3.predict(dtest_formatted)

labels = dtest_formatted.get_label()
print("preds are:", preds_1)
print("preds are:", preds_2)
print("preds are:", preds_3)
print("dtrainviews are:", dtrain_s.views[:100])

# print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))
# bst.save_model('0001.model')
# dump model
# bst.dump_model('dump.raw.txt')

