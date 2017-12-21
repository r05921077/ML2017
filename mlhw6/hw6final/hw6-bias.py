import numpy as np
import pandas as pd
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.models import Sequential
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.merge import Concatenate
from keras.layers.merge import Dot
import csv
#np.set_printoptions(threshold='nan')

df=pd.read_csv("users.csv", sep=',',header=None,encoding="big5")
age=[]
occ=[]
sex=[]
map= np.zeros((6041,3))
for row in range(1,6041):#899874
    tmp= df[0][row].split('::')
    map[int(tmp[0])][0]= int(tmp[2])
    map[int(tmp[0])][1]= int(tmp[3])
    if (tmp[1]=="M"):
        map[int(tmp[0])][2]=1
    else:
        map[int(tmp[0])][2]=0
user= np.load("./user.npy")
movie= np.load("./movie.npy")
rate= np.load("./rate.npy")
for a in user:
    age.append(map[a][0])
    occ.append(map[a][1])
    sex.append(map[a][2])
    #print(a,map[a][0],map[a][1])
age= np.array(age)
occ= np.array(occ)
sex= np.array(sex)
max_userid = user.max()+1
max_movieid = movie.max()+1
max_ageid = int(age.max()+1)
max_occid = int(occ.max()+1)
print(max_userid)
print(max_movieid)
print(user.shape)
print(movie.shape)
print(rate.shape)
print(age.shape)

model = Sequential()

k_factors= 20
P = Sequential()
P.add(Embedding(max_userid, k_factors, input_length=1))
P.add(Reshape((k_factors,)))
Q = Sequential()
Q.add(Embedding(max_movieid, k_factors, input_length=1))
Q.add(Reshape((k_factors,)))
W = Sequential()
W.add(Embedding(max_ageid, 1, input_length=1))
W.add(Reshape((1,)))
E = Sequential()
E.add(Embedding(max_occid, 1, input_length=1))
E.add(Reshape((1,)))
R = Sequential()
R.add(Embedding(2, 1, input_length=1))
R.add(Reshape((1,)))
a=Merge([P, Q], mode='dot', dot_axes=1)
b=Merge([a, W, E, R], mode = 'concat')
model.add(b)
model.add(Dense(1, activation = 'linear'))
def root_mean_squared_error(y_true, y_pred):
	return K.mean(K.square(y_pred - y_true), axis=-1)
model.summary()
model.compile(loss=root_mean_squared_error, optimizer='adamax')

earlystopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath='best.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             mode='min')
model.fit([user, movie, age, occ, sex],rate ,nb_epoch=1000, batch_size=500, validation_split=0.1, verbose=2,callbacks=[earlystopping,checkpoint])
emb= np.array(model.layers[0].get_weights())
print("---",emb)
emb= np.array(model.layers[1].get_weights())
print("---",emb)
K.clear_session()
