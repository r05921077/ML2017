import numpy as np
import pandas as pd

import csv
answer= np.load("./answer.npy")
print(answer.shape)
df=pd.read_csv("submission.csv", sep=',',header=None,encoding="big5")
pred=[]
for row in range(1,100337):#100337
    pred.append(float(df[1][row]))
    #print(answer[row-1],pred[row-1])


pua=pred[:50168]
pub=answer[:50168]

pra=pred[50168:]
prb=answer[50168:]


print("public:",np.sqrt(np.mean((pua-pub)**2)))
print("private:",np.sqrt(np.mean((pra-prb)**2)))