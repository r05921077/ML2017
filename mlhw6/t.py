import numpy as np
a=np.array([4,1,2,3,4])
b=np.where(a>2)
c=a[b]
for i in range(len(a)):
    if (a[i]>2):
        print(i)
        a[i]=99
print(a)