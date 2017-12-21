import numpy as np
import pandas as pd

import csv
#np.set_printoptions(threshold='nan')
'''
df=pd.read_csv("ratings.csv", sep=',',header=None,encoding="big5")
ck= -1
cnt= -1
user=[]
movie=[]
rate=[]
map= np.zeros((6050,3953))
for row in range(1,1000210):#899874
    k= np.fromstring(df[0][row],dtype=int,sep=' ')
    if ck != int(k[1]):
        ck= int(k[1])
        cnt += 1
        print(cnt,k[1])
    map[cnt][int(k[2])]= int(k[3])

np.save("./map",map)
'''
map= np.load("./map.npy")
user= np.load("./user.npy")
movie= np.load("./movie.npy")
rate= np.load("./rate.npy")
print(map.shape)


'''
map2= np.zeros((6050))
df=pd.read_csv("train.csv", sep=',',header=None,encoding="big5")
ck= -1
cnt= -1
for row in range(1,899874):#100337
    if ck != int(df[1][row]):
        ck= int(df[1][row])
        cnt += 1
        print(cnt,df[1][row])
    map2[int(df[1][row])]= cnt
np.save("./map2",map2)
'''
map2= np.load("./map2.npy")







result= []
df=pd.read_csv("test.csv", sep=',',header=None,encoding="big5")
ck= -1
cnt= -1
for row in range(1,100337):#100337
    if ck != int(df[1][row]):
        ck= int(df[1][row])
        cnt += 1
        print(cnt,df[1][row])
    a=int(map2[int(df[1][row])])
    b=int(df[2][row])
    result.append(map[a][b])
    print(map[a][b])
np.save("./answer",result)

'''
new=[]
with open("answer.csv","w", newline='') as f:###########輸出
    w = csv.writer(f)
    w.writerow(['answer'])
    for i in range (0, 100336):  #100336    
        print(i,result[i])         
        new.append(result[i])
        w.writerow(new)
    f.close()
'''