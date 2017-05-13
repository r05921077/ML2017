import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.neighbors import NearestNeighbors

np.set_printoptions(threshold=np.nan) 
pic=[]
for i in range(481):
	i += 1
	pic.append(plt.imread('.\hand\\hand.seq'+str(i)+'.png'))
pic= np.array(pic)
pic= np.reshape(pic, (481, 480*512))
print(pic.shape)


#########################################################################
def get_eigenvalues(data):
    SAMPLE = 20 # sample some points to estimate
    NEIGHBOR = 30 # pick some neighbor to compute the eigenvalues
    randidx = np.random.permutation(data.shape[0])[:SAMPLE]
    knbrs = NearestNeighbors(n_neighbors=NEIGHBOR,
                             algorithm='ball_tree').fit(data)

    sing_vals = []
    for idx in randidx:
        dist, ind = knbrs.kneighbors(data[idx:idx+1])
        nbrs = data[ind[0,1:]]
        print("a")
        u, s, v = np.linalg.svd(nbrs - nbrs.mean(axis=0), full_matrices=False)
        print("b")
        s /= s.max()
        sing_vals.append(s)
    sing_vals = np.array(sing_vals).mean(axis=0)
    return sing_vals


vs = get_eigenvalues(pic)
print(vs.shape)
print(vs)


