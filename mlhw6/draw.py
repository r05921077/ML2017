import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import csv
y=[]
y=np.array(y)
a= np.load("./emb.npy")
a= np.reshape(a, (3953, 20))
print(a.shape)
df=pd.read_csv("movies.csv",  sep='\t',header=None,encoding="latin-1")

tags=[]
item=[]
rtags=np.zeros((3953))
for row in range(1,3884):#899874
	tmp= df[0][row].split('::')
	id= int(tmp[0])
	tmp= tmp[2].split('|')
	tmp= np.random.choice(tmp)
	if (  tmp.find("Drama")>-1 or tmp.find("Musical") >-1):
		tags.append(500)
	elif (  tmp.find("Thriller")>-1 or tmp.find("Horror")>-1 or tmp.find("Crime")>-1 ):
		tags.append(600)
	elif (  tmp.find("Adventure")>-1 or tmp.find("Animation")>-1 or tmp.find("Children's")>-1):
		tags.append(700)
	else:
		if tmp not in item:
			item.append(tmp)
			tags.append(len(item)*50-1)
		else:
			for i in range(len(item)):
				if tmp == item[i]: 
					tags.append(i*50)
	rtags[id]=tags[row-1]
	print(id,tags[row-1])
tags= np.array(tags)
for i in range(len(item)):
	print(item[i],i*50)
model2 = TSNE(n_components=2, random_state=0) ### TSNE
np.set_printoptions(suppress=True)
#vis_data= model2.fit_transform(a)
#np.save('./tsne.npy', vis_data)
vis_data= np.load('./tsne.npy')
print(vis_data.shape)
print(tags.shape)

vis_x= vis_data[:, 0]
vis_y= vis_data[:, 1]

cm= plt.cm.get_cmap('RdYlBu')
sc= plt.scatter(vis_x, vis_y, c=rtags, cmap=cm)
plt.colorbar(sc)
plt.show()

print(a)