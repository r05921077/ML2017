import word2vec as w
import numpy as np
import nltk
from sklearn.manifold import TSNE
from adjustText import adjust_text
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)  ### word2vec
w.word2vec('./all.txt', './text8.bin', size=50, verbose=True)
model = w.load('./text8.bin')
print("\n\n")

#word= "know"
#print(model[word].shape)
#a=model[word]
#print(a)
#indexes, metrics = model.cosine(word)
#print(model.generate_response(indexes, metrics))


################################################ Visualization of Word Vectors
size= 500
vocab= model.vocab[:size]
vector= model.vectors[:size,:]
pvocab= []
pvector= []

model2 = TSNE(n_components=2, random_state=0) ### TSNE
np.set_printoptions(suppress=True)
vector= model2.fit_transform(vector)

vocab= nltk.pos_tag(vocab)### NTLK
cnt= 0
for i in range(size):
	if (vocab[i][1] == "JJ" or vocab[i][1] == "NN" or vocab[i][1] == "NNP" or vocab[i][1] == "NNS") and (len(vocab[i][0]) != 1) and "." not in vocab[i][0] and "|" not in vocab[i][0] and "“" not in vocab[i][0] and "’" not in vocab[i][0] and "," not in vocab[i][0] and "“" not in vocab[i][0] and "?" not in vocab[i][0]:
		pvocab.append(vocab[i])
		pvector.append(vector[i])
		cnt += 1
pvector= np.array(pvector)
pvector= pvector.T

plt.plot(pvector[0,],pvector[1,],'ro')### Plot
texts= []
n= pvector.shape[1]
for i in range(n):
	texts.append(plt.text(pvector[0][i],pvector[1][i],pvocab[i][0]))
adjust_text(texts, arrowprops=dict(arrowstyle="->", color='b', lw=0.5))
plt.show()

