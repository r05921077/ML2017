from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras import optimizers
import tensorflow as tf
import numpy as np
import csv
import pandas as pd
from keras import backend as K
from keras import regularizers
import keras.utils
from keras.models import model_from_json
import sys

with tf.device('/gpu:0'):
    json_file = open('STRONG2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("STRONG2.h5")
    print("Loaded model from disk")
 
    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("--------------------training over--------------------")
df=pd.read_csv(sys.argv[1], sep=',',header=None,encoding="big5")
data = []
for row in range(1,7179):
    feat = np.fromstring(df[1][row],dtype=int,sep=' ')
    feat = np.reshape(feat,(48,48,1))
    data.append(feat)
x= np.array(data)
result= loaded_model.predict(x)
print(result,result.shape)

with open(sys.argv[2],"w", newline='') as f:###########輸出
    w = csv.writer(f)
    w.writerow(['id','label'])
    for i in range (0, 7178):  
        a=str(i)
        new = []                 
        new.append(a)
        qq= 0 
        for j in range (0,7):
            if result[i][j]>qq:
                max= j
                qq= result[i][j]
        new.append(str(max))
        w.writerow(new)
    f.close()

K.clear_session()
