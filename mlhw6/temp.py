import numpy as np
import pandas as pd
import keras
#np.set_printoptions(threshold='nan')
df=pd.read_csv("train.csv", sep=',',header=None,encoding="big5")
data = np.zeros((6040, 3952))
ck= -1
cnt= -1
for row in range(1,899874):#899874
    if ck != int(df[1][row]):
        cnt += 1
        ck= int(df[1][row])
        print(cnt,df[1][row])
    #print(row)
    data[cnt][int(df[2][row])-1]= int(df[3][row])


P = Sequential()
P.add(Embedding(6040, 2, input_length=1))
P.add(Reshape((k_factors,)))
Q = Sequential()
Q.add(Embedding(3952, 2, input_length=1))
Q.add(Reshape((k_factors,)))
super(DeepModel, self).__init__(**kwargs)
self.add(Merge([P, Q], mode='concat'))
self.add(Dropout(p_dropout))
self.add(Dense(k_factors, activation='relu'))
self.add(Dropout(p_dropout))
self.add(Dense(1, activation='linear'))
model.compile(loss='rmse', optimizer='adamax')
model.fit([Users, Movies], Ratings, nb_epoch=30, validation_split=.1, verbose=2, callbacks=callbacks)

print(data)