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

column_labels.remove("num_private")
column_labels.remove("recorded_by")
column_labels.remove("wpt_name")#
column_labels.remove("extraction_type_group")
column_labels.remove("extraction_type")#
column_labels.remove("payment_type")
column_labels.remove("water_quality")#
column_labels.remove("scheme_management")#
column_labels.remove("district_code")#
column_labels.remove("region")
column_labels.remove("region_code")#
column_labels.remove("subvillage")#
column_labels.remove("ward")#
column_labels.remove("waterpoint_type_group")
column_labels.remove("quantity_group")
column_labels.remove("installer")

#column_labels.remove("num_private")Accuracy = 0.815488215488
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

k=train["construction_year"]
k=np.array(k)
k=k-1960
id=np.where(k>0)
tmp=k[id]
mid=np.median(tmp)
for i in range(len(k)):
    if(k[i]==0):
        k[i]= mid
train["construction_year"]= k
#print("constru mid=",mid)
#for i in k:
#    print(i,end=",")
k=train["gps_height"]
k=np.array(k)
id=np.where(k>0)
tmp=k[id]
mid=np.median(tmp)
for i in range(len(k)):
    if(k[i]==0):
        k[i]= mid
train["gps_height"]= k
#print("gps mid=",mid)

np.random.seed(seed=580)#0.817508417508
train = train.iloc[np.random.permutation(len(train))]
#np.random.seed(seed=580)#580:0.816077441077    811111:0.817171717172
# Assign data for validation
amount = int(0.8*len(train))
validation = train[int(amount):]
#train = train[:amount]
print("Assign data for validation: successfully")

# Classifier
# clf = tree.DecisionTreeClassifier()
clf = RandomForestClassifier(n_estimators = 200, n_jobs = -1,oob_score=True)#0.812205387205


print("Classifier: successfully")

# Traning
clf.fit(train[column_labels], train["status_group"])
print("Traning: successfully")

# Accuracy
accuracy = accuracy_score(clf.predict(validation[column_labels]), validation["status_group"])
print("Accuracy = " + str(accuracy))
print("Accuracy: successfully")
#print("oob score:",clf.oob_score_ )

# Free some ram
del train, validation
gc.collect()


###Testing part###
# Testing data
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

k=test["construction_year"]
k=np.array(k)
k=k-1960
id=np.where(k>0)
tmp=k[id]
mid=np.median(tmp)
for i in range(len(k)):
    if(k[i]<0):
        k[i]= mid
test["construction_year"]= k
print("Testing data: successfully")
# Prediction for test data
prediction = clf.predict(test[column_labels])
print("Prediction for test data: successfully")

k=test["gps_height"]
k=np.array(k)
id=np.where(k>0)
tmp=k[id]
mid=np.median(tmp)
for i in range(len(k)):
    if(k[i]==0):
        k[i]= mid
test["gps_height"]= k


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
