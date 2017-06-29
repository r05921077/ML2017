
import pandas as pd
import xgboost as xgb
import time
import numpy as np


###Training part###
print('read Training Data')
dataset = pd.read_csv("train_data.csv")

dataset.drop(dataset.columns[[0,2]], axis=1, inplace=True)

labels = dataset['status_group']
dataset.pop('status_group')
train = dataset.iloc[:,:].values
# Features for trainig


status_group = ["functional", "non functional", "functional needs repair"]


###Testing part###
# Testing data
tests = pd.read_csv("test.csv")
id = tests['id']
tests.drop(tests.columns[[0,2]], axis=1, inplace=True)

test = tests.iloc[:,:].values


offset = 50000
xgtrain = xgb.DMatrix(train[:offset,:], label=labels[:offset])
xgval = xgb.DMatrix(train[offset:,:], label=labels[offset:])
xgtest = xgb.DMatrix(test)

params={
'booster':'gbtree',
'objective': 'multi:softmax',
'num_class':3,
'gamma':0.01,
'max_depth’:30,
'subsample':1,
'colsample_bytree':0.4,
'silent':1 ,
'eta': 0.05,
'seed':710,
'nthread':2,
'eval_metric':'merror'
}

plst = list(params.items())
num_rounds = 93

evallist  = [(xgval,'eval'), (xgtrain,'train')]
model = xgb.train(plst, xgtrain , num_rounds,evallist)

# Prediction for test data
prediction = model.predict(xgtest,ntree_limit=model.best_iteration)
print("Prediction for test data: successfully")


### Making submission file###
# Dataframe as per submission format
submission = pd.DataFrame({
                          "id": id,
                          "status_group": prediction
                          })
for i in range(len(status_group)):
    submission.loc[submission["status_group"] == i, "status_group"] = status_group[i]
print("Dataframe as per submission format: successfully")

# Store submission dataframe into file
submission.to_csv("submission.csv", index = False)
print("Store submission dataframe into file: successfully")
