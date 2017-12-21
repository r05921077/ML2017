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

for a in user:
    age.append(map[a][0])
age= np.array(age)
max_userid = user.max()+1
max_movieid = movie.max()+1
max_ageid = int(age.max()+1)

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
r_hat= Dot(axes=1)([user_vec, item_vec])
r_hat= Add()([r_hat, user_bias])
model= Model([user_input, item_input,bias_input], r_hat)
def root_mean_squared_error(y_true, y_pred):
	return K.mean(K.square(y_pred - y_true), axis=-1)
model.summary()
model.load_weights('hw6.hdf5')
model.compile(loss=root_mean_squared_error, optimizer='adamax')


df=pd.read_csv("test.csv", sep=',',header=None,encoding="big5")
ck= -1
cnt= -1
user=[]
movie=[]
age=[]
for row in range(1,100337):#100337
    if ck != int(df[1][row]):
        ck= int(df[1][row])
        cnt += 1
        print(cnt,df[1][row])
    user.append(int(df[1][row]))
    movie.append(int(df[2][row]))
    age.append(map[int(df[1][row])][0])
user= np.array(user)
movie= np.array(movie)
age= np.array(age)
result= model.predict([user, movie, age])
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
