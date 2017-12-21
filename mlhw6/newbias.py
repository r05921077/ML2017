import numpy as np
import pandas as pd
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.models import Sequential
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.merge import Concatenate
from keras.layers.merge import Dot
import csv
from keras.layers import Input, Flatten, Add
from keras.models import Model 
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
print(movie)
model = Sequential()

k_factors= 120
user_input= Input(shape=[1])
item_input= Input(shape=[1])
bias_input= Input(shape=[1])
user_vec= Embedding(max_userid,k_factors)(user_input)
user_vec= Flatten()(user_vec)
item_vec= Embedding(max_movieid,k_factors)(item_input)
item_vec= Flatten()(item_vec)
user_bias= Embedding(max_userid,1)(bias_input)
user_bias= Flatten()(user_bias)
#item_bias= Embedding(max_userid,1)(user_input)
#item_bias= Flatten()(item_bias)
r_hat= Dot(axes=1)([user_vec, item_vec])
r_hat= Add()([r_hat, user_bias])
model= Model([user_input, item_input,bias_input], r_hat)
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
model.fit([user, movie,age],rate ,nb_epoch=1000, batch_size=500, validation_split=0.1, verbose=2,callbacks=[earlystopping,checkpoint])
emb= np.array(model.layers[0].get_weights())
print("---",emb)
emb= np.array(model.layers[1].get_weights())
print("---",emb)
K.clear_session()
