#!/usr/bin/env python
# -- coding: utf-8 --

import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.models import Sequential
import pandas as pd
import keras.utils
from statsmodels.iolib.smpickle import load_pickle as load
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return abs(x / (K.sqrt(K.mean(K.square(x))) + 1e-5))

y= []
x= []
df=pd.read_csv('train.csv', sep=',',header=None,encoding="big5")
for row in range(1,1001):
    y.append(int(df[0][row]))
    feat = np.fromstring(df[1][row],dtype=int,sep=' ')
    feat = np.reshape(feat,(48,48,1))
    x.append(feat)
    k=[int(x.strip()) for x in df[1][row].split(' ')]
x= np.array(x)
y= np.array(y)
y = keras.utils.to_categorical(y, 7)
model = Sequential()
print("ok")

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
model_dir = os.path.join(base_dir, 'model')





parser = argparse.ArgumentParser(prog='plot_saliency.py',
        description='ML-Assignment3 visualize attention heat map.')
parser.add_argument('--epoch', type=int, metavar='<#epoch>', default=80)
args = parser.parse_args()
np.set_printoptions(threshold=np.nan)
json_file = open('STRONG2.json', 'r')
emotion_classifier = json_file.read()
json_file.close()
loaded_model = model_from_json(emotion_classifier)
# load weights into new model
loaded_model.load_weights("STRONG2.h5")
print("Loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

private_pixels = x
#private_pixels = [ np.fromstring(private_pixels[i], dtype=float, sep=' ').reshape((1, 48, 48, 1)) 
#                       for i in range(len(private_pixels)) ]
input_img = loaded_model.input
img_ids = ["image ids from which you want to make heatmaps"]

for idx in range (50):
    print(idx)
    pred = loaded_model.predict_classes(np.array([private_pixels[idx]]))
    target = K.mean(loaded_model.output[:, pred])
    grads = K.gradients(target, input_img)[0]
    grads = normalize(grads)
    fn = K.function([input_img, K.learning_phase()], [grads])

    step= 1
    #heatmap = private_pixels[idx].reshape(48, 48)
    heatmap = np.zeros((1,48,48,1))
    heatmap= np.array(heatmap, dtype=float)

    for i in range(1):
        grads_value= np.array(fn([np.array([private_pixels[idx]]),0]))
        heatmap = grads_value[0] 
    
    '''
    Implement your heatmap processing here!
    hint: Do some normalization or smoothening on grads
    '''

    thres = 0.5
    see = private_pixels[idx].reshape(48, 48)
    heatmap.reshape(48, 48)
    heatmap2 = np.zeros((48,48))
    for i in range(48):
        for j in range(48):
            heatmap2[i][j]= heatmap[0][i][j][0]

    see[np.where(heatmap2 <= thres)] = np.mean(see)
    print(heatmap2[20])
    print(see[20])
    
    plt.figure()
    plt.imshow(heatmap2, cmap=plt.cm.jet)
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig('x%d.png' % (idx), dpi=100)

    plt.figure()
    plt.imshow(see,cmap='gray')
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig('y%d.png' % (idx), dpi=100)

