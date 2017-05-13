import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
def draw(U,n,m):
	global mean
	U= U.T
	U= np.reshape(U,(100, 64, 64))
	if m:
		U= U+mean
	margin= 2
	width = n * 64 + (n - 1) * margin
	height = n * 64 + (n - 1) * margin
	eigface = np.zeros((width, height))
	for i in range(n):
	    for j in range(n):
	        eigface[(64 + margin) * i: (64 + margin) * i + 64,
	                         (64 + margin) * j: (64 + margin) * j + 64] = U[i*n+j]
	plt.imshow(eigface,cmap='gray')
	plt.show()

face= []
for j in range(10):
	for i in range(10):
		k= misc.imread('.\pic\\'+chr(65+j)+'0'+str(i)+'.bmp')
		face.append(k)
face= np.array(face)
print(face.shape)
face= np.reshape(face,(100, 64*64))
mean= face.mean(axis=0)
face= face - mean
face= face.T
mean= np.reshape(mean,(64,64))
print(face.shape, face.dtype)
############################################### 抓完圖片

U, s, V = np.linalg.svd(face, full_matrices=False)### SVD
S = np.diag(s)

plt.imshow(mean,cmap='gray')### P1
plt.show()
draw(U, 3, 0)

size= 5### P2
recst= np.dot(U[:, :size], np.dot(S[:size, :size], V[:size,:]))
draw(recst, 10, 1)
draw(face, 10, 1)

size= 59### P3
recst= np.dot(U[:, :size], np.dot(S[:size, :size], V[:size,:]))
print("Error=",(np.sqrt(np.mean((face - recst)**2))/256))
