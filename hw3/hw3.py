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
from keras.callbacks import Callback
import sys

class History(Callback):

    def on_epoch_end(self,epoch,logs={}):
    	if logs.get('val_acc')>0.99:
            self.model.stop_training = True

y= []
x= []
rx= []
valid=[]
vy=[]
df=pd.read_csv(sys.argv[1], sep=',',header=None,encoding="big5")
for row in range(1,28710):
	feat = np.fromstring(df[1][row],dtype=int,sep=' ')
	feat = np.reshape(feat,(48,48,1))
	temp=  np.reshape(feat,(48,48))
	temp= np.fliplr(temp)
	temp= np.reshape(temp,(48,48,1))
	if row<28710-2871:
		x.append(feat)
		rx.append(temp)
		y.append(int(df[0][row]))
	else:
		valid.append(feat)
		valid.append(temp)
		vy.append(int(df[0][row]))
		vy.append(int(df[0][row]))
	k=[int(x.strip()) for x in df[1][row].split(' ')]

x= np.array(x)
rx= np.array(rx)
x=np.concatenate((x, rx), axis=0)
y=np.concatenate((y, y), axis=0)
valid= np.array(valid)
vy= np.array(vy)
x=np.concatenate((x, valid), axis=0)
y=np.concatenate((y, vy), axis=0)
print(x.shape)
print(valid.shape)
print(y.shape)
print(vy.shape)
y= np.array(y)
y = keras.utils.to_categorical(y, 7)
model = Sequential()
print("ok")

drop=0.5
with tf.device('/gpu:0'):
	model.add(Conv2D(32,(3,3),input_shape=(48,48,1),data_format='channels_last',padding= 'same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(drop))	
	model.add(Conv2D(64,(3,3),data_format='channels_last',padding= 'same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64,(3,3),data_format='channels_last',padding= 'same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(drop))

	model.add(Conv2D(128,(3,3),data_format='channels_last',padding= 'same'))
	model.add(Activation('relu'))
	model.add(Conv2D(128,(3,3),data_format='channels_last',padding= 'same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(drop))	
	model.add(Conv2D(256,(3,3),data_format='channels_last',padding= 'same'))
	model.add(Activation('relu'))
	model.add(Conv2D(256,(3,3),data_format='channels_last',padding= 'same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(drop))


	model.add(Flatten())

	reg=0.00
	model.add(Dense(output_dim=128,kernel_regularizer=regularizers.l2(0.01)))
	model.add(Activation('relu'))
	model.add(Dropout(drop))
	model.add(Dense(output_dim=128,kernel_regularizer=regularizers.l2(0.01)))
	model.add(Activation('relu'))
	model.add(Dropout(drop))
	
	model.add(Dense(output_dim=7, kernel_regularizer=regularizers.l2(0.0)))
	model.add(Activation('softmax'))
	model.summary()

	sg = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0, nesterov=True) 
	opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)
	model.compile(loss='categorical_crossentropy',
	              optimizer= 'adam',
	              metrics=['accuracy'])
	history = History()
	model.fit(x, y, epochs=300, batch_size=1000,  validation_split=0.1, callbacks=[history])

	print("--------------------training over--------------------")

K.clear_session()
