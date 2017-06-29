import gc
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

###Training part###
# Traning data
train = pd.read_csv("train_data.csv")
print("Traning data: successfully")

# Features for trainig
column_labels = list(train.columns.values)
'''
column_labels.remove("num_private")
column_labels.remove("recorded_by")
#column_labels.remove("wpt_name")#
column_labels.remove("extraction_type_group")
#column_labels.remove("extraction_type")#
column_labels.remove("payment_type")
#column_labels.remove("water_quality")#
#column_labels.remove("scheme_management")#
#column_labels.remove("district_code")#
column_labels.remove("region")
#column_labels.remove("region_code")#
#column_labels.remove("subvillage")#
#column_labels.remove("ward")#
column_labels.remove("waterpoint_type_group")
column_labels.remove("quantity_group")
column_labels.remove("installer")
'''
column_labels.remove("id")
column_labels.remove("status_group")
status_group = ["functional", "non functional", "functional needs repair"]
print("Features for trainig: successfully")
k=train["date_recorded"]
qq=[]
for j in range(len(k)):
    k=np.array(k)
    a=k[j].find('-')
    b= k[j].find('-',a+1)
    leng=len(k[j])
    tmp=k[j][a-1]
    for i in range(a+1,b):
        if (b-a==2):
            tmp=tmp+'0'
        tmp=tmp+k[j][i]
    for i in range(b+1,leng):
        if (leng-b==2):
            tmp=tmp+'0'
        tmp=tmp+k[j][i]
    qq.append(tmp)
print(len(qq))
train["date_recorded"]=qq
#print(train)
np.random.seed(seed=580)
train = train.iloc[np.random.permutation(len(train))]



test = pd.read_csv("test.csv")
test = test.fillna(test.median())

k=test["date_recorded"]
qq=[]
for j in range(len(k)):
    k=np.array(k)
    a=k[j].find('-')
    b= k[j].find('-',a+1)
    leng=len(k[j])
    tmp=k[j][a-1]
    for i in range(a+1,b):
        if (b-a==2):
            tmp=tmp+'0'
        tmp=tmp+k[j][i]
    for i in range(b+1,leng):
        if (leng-b==2):
            tmp=tmp+'0'
        tmp=tmp+k[j][i]
    qq.append(tmp)
print(len(qq))
test["date_recorded"]=qq
print("Testing data: successfully")
###############################################################3333###############################################################3333

testx=test[column_labels]
testx=np.array(testx)
trainx=train[column_labels]
trainx=np.array(trainx)
trainy=train["status_group"]
trainy=np.array(trainy)


amount = int(0.8*len(trainx))
validationx = trainx[int(amount):]
validationy = trainy[int(amount):]
#trainx = trainx[:amount]
#trainy = trainy[:amount]
for i in range (2):
    # Assign data for validation
    print("\nSelf loop:",i+1)
    np.random.seed(seed=580)
    train.iloc[np.random.permutation(len(train))]
    print("Training size:",len(trainx))
    # Classifier
    clf = RandomForestClassifier(n_estimators = 200, n_jobs = -1)
    clf = RandomForestClassifier(criterion='gini',
                                min_samples_split=8,
                                n_estimators=1000,
                                max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)
    # Traning
    clf.fit(trainx, trainy)
    # Accuracy
    accuracy = accuracy_score(clf.predict(validationx), validationy)
    print("Accuracy = " + str(accuracy))
    # Self training
    ans= clf.predict(testx)
    prob= clf.predict_proba(testx)
    add= np.where(prob > 0.65)
    print("added data:",len(add[0]))
    selfx= testx[add[0]]
    selfy= ans[add[0]]
    testx= np.delete(testx, add[0], axis = 0)
    trainx = np.concatenate((trainx, selfx))
    trainy = np.concatenate((trainy, selfy))


###############################################################3333###############################################################3333










prediction = clf.predict(test[column_labels])
print("Prediction for test data: successfully")
### Making submission file###
# Dataframe as per submission format
submission = pd.DataFrame({
			"id": test["id"],
			"status_group": prediction
		})
for i in range(len(status_group)):
	submission.loc[submission["status_group"] == i, "status_group"] = status_group[i]
print("Dataframe as per submission format: successfully")

# Store submission dataframe into file
submission.to_csv("submission.csv", index = False)
print("Store submission dataframe into file: successfully")


del trainx, validationx,trainy, validationy
gc.collect()