#!/usr/bin/env python
# -- coding: utf-8 --

from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
import itertools


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


valid=[]
vy=[]
fear= []
cnt= 0
df=pd.read_csv('train.csv', sep=',',header=None,encoding="big5")
for row in range(28710-2871,28710):
    feat = np.fromstring(df[1][row],dtype=int,sep=' ')
    feat = np.reshape(feat,(48,48,1))
    temp=  np.reshape(feat,(48,48))
    temp= np.reshape(temp,(48,48,1))
    valid.append(feat)
    vy.append(int(df[0][row]))



valid= np.array(valid)
vy= np.array(vy)
print(valid.shape)
print(vy.shape)
model = Sequential()
print("ok")

np.set_printoptions(threshold=np.nan)
json_file = open('STRONG2.json', 'r')
emotion_classifier = json_file.read()
json_file.close()
loaded_model = model_from_json(emotion_classifier)
# load weights into new model
loaded_model.load_weights("STRONG2.h5")
print("Loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            
predictions = loaded_model.predict_classes(valid)
pred = loaded_model.predict(valid)
qq=np.array([0,0,0,0,0,0,0],dtype= float)
for i in range(2871):
    if int(vy[i]) == 2:
        cnt += 1
        fear.append(pred[i])
        for j in range(7):
            qq[j] += pred[i][j]
for i in fear:
    print(i)
print("yeeah")
print(cnt)
print(qq/cnt)
conf_mat = confusion_matrix(vy,predictions)
plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.show()
print("yoyo")