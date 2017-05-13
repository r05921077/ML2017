import numpy as np
from sklearn.svm import LinearSVR as SVR
from sklearn.neighbors import NearestNeighbors
import csv
import gen as gg
np.set_printoptions(threshold=np.nan)

def get_eigenvalues(data):
    SAMPLE = 100 # sample some points to estimate
    NEIGHBOR = 300 # pick some neighbor to compute the eigenvalues
    randidx = np.random.permutation(data.shape[0])[:SAMPLE]
    knbrs = NearestNeighbors(n_neighbors=NEIGHBOR,
                             algorithm='ball_tree').fit(data)

    sing_vals = []
    for idx in randidx:
        dist, ind = knbrs.kneighbors(data[idx:idx+1])
        nbrs = data[ind[0,1:]]
        u, s, v = np.linalg.svd(nbrs - nbrs.mean(axis=0), full_matrices=False)
        s /= s.max()
        sing_vals.append(s)
    sing_vals = np.array(sing_vals).mean(axis=0)
    return sing_vals

# Train a linear SVR

X = gg.X
y = gg.y

print("training start")
svr = SVR(C=5)
svr.fit(X, np.log(y))
print("training end")


testdata = np.load('data.npz') ########### predict
test_X = []
for i in range(200):
    print(i)
    data = testdata[str(i)]
    vs = get_eigenvalues(data)
    vs= np.append(vs, np.std(testdata[str(i)]))
    test_X.append(vs)

test_X = np.array(test_X)
result = svr.predict(test_X)

with open("submission.csv","w", newline='') as f:########### output
	w = csv.writer(f)
	w.writerow(['SetId','LogDim'])
	for i in range (0, 200):  
		a=str(i)
		new = []                 
		new.append(a)
		qq= 0 
		if result[i]<0:
			result[i]=0
		if result[i]>4.09434456:
			result[i]=4.09434456
		new.append(str(result[i]))
		w.writerow(new)
	f.close()
print("done")

