import numpy as np
import string
import sys
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from six.moves import cPickle
import sys
import os
from keras.models import load_model
from keras.layers.wrappers import Bidirectional


test_path = sys.argv[1]
output_path = sys.argv[2]

#####################
###   parameter   ###
#####################
split_ratio = 0.1
embedding_dim = 100
nb_epoch = 1000
batch_size = 128


################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r', encoding = 'utf8') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

###########################
###   custom metrices   ###
###########################
def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

#########################
###   Main function   ###
#########################

### read training and testing data
(_, X_test,_) = read_data(test_path,False)
tag_list=['SCIENCE-FICTION', 'SPECULATIVE-FICTION', 'FICTION', 'NOVEL', 'FANTASY', "CHILDREN'S-LITERATURE", 'HUMOUR', 'SATIRE', 'HISTORICAL-FICTION', 'HISTORY', 'MYSTERY', 'SUSPENSE', 'ADVENTURE-NOVEL', 'SPY-FICTION', 'AUTOBIOGRAPHY', 'HORROR', 'THRILLER', 'ROMANCE-NOVEL', 'COMEDY', 'NOVELLA', 'WAR-NOVEL', 'DYSTOPIA', 'COMIC-NOVEL', 'DETECTIVE-FICTION', 'HISTORICAL-NOVEL', 'BIOGRAPHY', 'MEMOIR', 'NON-FICTION', 'CRIME-FICTION', 'AUTOBIOGRAPHICAL-NOVEL', 'ALTERNATE-HISTORY', 'TECHNO-THRILLER', 'UTOPIAN-AND-DYSTOPIAN-FICTION', 'YOUNG-ADULT-LITERATURE', 'SHORT-STORY', 'GOTHIC-FICTION', 'APOCALYPTIC-AND-POST-APOCALYPTIC-FICTION', 'HIGH-FANTASY']
print(tag_list)
### tokenizer for all data
tokenizer = cPickle.load(open(os.path.join("./", "hw5.pkl"), 'rb'))
word_index = tokenizer.word_index
### convert word sequences to index sequence
print ('Convert to index sequences.')
test_sequences = tokenizer.texts_to_sequences(X_test)
### padding to equal length
print ('Padding sequences.')
max_article_length = 306
test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)


num_words = len(word_index) + 1
### build model
print ('Building model.')
model = Sequential()
model.add(Embedding(num_words,
                    embedding_dim,
                    input_length=max_article_length,
                    trainable=False))
drop= 0.25
#model.add(GRU(128, return_sequences=True,activation='tanh',dropout=0.25))
model.add(GRU(128,activation='tanh',dropout=0.25))
model.add(Dense(256,activation='relu'))
model.add(Dropout(drop))
model.add(Dense(128,activation='relu'))
model.add(Dropout(drop))
model.add(Dense(64,activation='relu'))
model.add(Dropout(drop))
model.add(Dense(38,activation='sigmoid'))
model.summary()
adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
model.load_weights('hw5.hdf5')
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=[f1_score])

Y_pred = model.predict(test_sequences)
thresh = 0.4
with open(output_path,'w') as output:
    print ('\"id\",\"tags\"',file=output)
    Y_pred_thresh = (Y_pred > thresh).astype('int')
    for index,labels in enumerate(Y_pred_thresh):
        labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
        labels_original = ' '.join(labels)
        print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

print("\nend")
K.clear_session()