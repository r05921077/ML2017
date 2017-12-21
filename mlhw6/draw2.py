import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import csv
y=[]
y=np.array(y)
a= np.load("./emb2.npy")
a= np.reshape(a, (6041, 20))
print(a[0])
tags=np.zeros((6041))
df=pd.read_csv("users.csv", sep=',',header=None,encoding="big5")
for row in range(1,6041):#899874
    tmp= df[0][row].split('::')
    tags[int(tmp[0])-1]=tmp[3]

tags= np.array(tags)
for i in tags:
	print(i)
model2 = TSNE(n_components=2, random_state=0) ### TSNE
np.set_printoptions(suppress=True)
#vis_data= model2.fit_transform(a)
#np.save('./tsne2.npy', vis_data)
vis_data= np.load('./tsne2.npy')
print(vis_data.shape)
print(tags.shape)

vis_x= vis_data[:, 0]
vis_y= vis_data[:, 1]

cm= plt.cm.get_cmap('RdYlBu')
sc= plt.scatter(vis_x, vis_y, c=tags, cmap=cm)
plt.colorbar(sc)
plt.show()

print(a)