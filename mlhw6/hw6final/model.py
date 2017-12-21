import numpy as np
import pandas as pd
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.models import Sequential
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
import csv

user= np.load("./user.npy")
movie= np.load("./movie.npy")
rate= np.load("./rate.npy")

max_userid = user.max()+1
max_movieid = movie.max()+1

model = Sequential()

k_factors= 120
P = Sequential()
P.add(Embedding(max_userid, k_factors, input_length=1))
P.add(Reshape((k_factors,)))
Q = Sequential()
Q.add(Embedding(max_movieid, k_factors, input_length=1))
Q.add(Reshape((k_factors,)))
model.add(Merge([P, Q], mode='dot', dot_axes=1))
def root_mean_squared_error(y_true, y_pred):
	return K.mean(K.square(y_pred - y_true), axis=-1)
model.load_weights('120.hdf5')
model.compile(loss=root_mean_squared_error, optimizer='adamax')


df=pd.read_csv("test.csv", sep=',',header=None,encoding="big5")
ck= -1
cnt= -1
user=[]
movie=[]
for row in range(1,100337):#100337
    if ck != int(df[1][row]):
        ck= int(df[1][row])
        cnt += 1
        print(cnt,df[1][row])
    user.append(int(df[1][row]))
    movie.append(int(df[2][row]))
user= np.array(user)
movie= np.array(movie)
result= model.predict([user, movie])
print(result.shape)


with open("submission.csv","w", newline='') as f:###########輸出
	w = csv.writer(f)
	w.writerow(['TestDataID','Rating'])
	for i in range (0, 100336):  #100336
		a=str(i+1)
		new = []                 
		new.append(a)
		if (result[i][0]<1):
			new.append(1)
		elif (result[i][0]>5):
			new.append(5)
		else:
			new.append(result[i][0])
		w.writerow(new)
	f.close()

K.clear_session()
