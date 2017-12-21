from __future__ import division
import numpy as np
import pandas as pd
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.models import Sequential
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
import csv
#np.set_printoptions(threshold='nan')
"""
df=pd.read_csv("train.csv", sep=',',header=None,encoding="big5")
ck= -1
cnt= -1
user=[]
movie=[]
rate=[]
for row in range(1,899874):#899874
    if ck != int(df[1][row]):
        ck= int(df[1][row])
        cnt += 1
        print(cnt,df[1][row])
    #print(row)
    data[cnt][int(df[2][row])-1]= int(df[3][row])
    user.append(int(df[1][row]))
    movie.append(int(df[2][row]))
    rate.append(int(df[3][row]))
np.random.seed(seed=1)
user= np.array(user)
np.random.shuffle(user)
np.random.seed(seed=1)
movie= np.array(movie)
np.random.shuffle(movie)
np.random.seed(seed=1)
rate= np.array(rate)
np.random.shuffle(rate)
np.save("./user",user)
np.save("./movie",movie)
np.save("./rate",rate)
"""
user= np.load("./user.npy")
movie= np.load("./movie.npy")
rate= np.load("./rate.npy")
rate=np.array(rate,dtype=float)
a=np.array(rate)
std=np.std(a,axis=0)
print(std)
mean=np.mean(a,axis=0)
print(mean)
for i in range(len(rate)):        
    rate[i]=(rate[i]-mean)/std

max_userid = user.max()+1
max_movieid = movie.max()+1
print(max_userid)
print(max_movieid)
print(user.shape)
print(movie.shape)
print(rate.shape)

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
model.compile(loss=root_mean_squared_error, optimizer='adamax')


earlystopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath='best.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             mode='min')
model.fit([user, movie],rate ,nb_epoch=1000, batch_size=500, validation_split=0.1, verbose=2,callbacks=[earlystopping,checkpoint])

K.clear_session()
