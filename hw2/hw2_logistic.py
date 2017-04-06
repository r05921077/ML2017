# -*- coding: utf-8 -*- 
import csv
from numpy import *
import numpy as np
import pandas as pd
import math
from numpy import genfromtxt
import sys

feat= 106
id1= 8500
lamb=1
b= -0.7
data = []       ########### original data                            
for i in range (0, 32561):                
	new = []                 
	for j in range (0, 106):    
		new.append('0')     
	data.append(new)
print("ok")
valid = []       ########### arrays                             
for i in range (0, 32561):                  
	new = []                 
	for j in range (0, feat):    
		new.append(0)     
	valid.append(new)

twod = []       ########### arrays                             
for i in range (0, id1):                  
	new = []                 
	for j in range (0, feat):    
		new.append(0)     
	twod.append(new)
y = []                                       
for i in range (0, id1):                                       
	y.append(0)  
w= []                                       
for i in range (0, feat):                                       
	w.append(0.00000000000000000001)
temp= []                                       
for i in range (0, feat):                                       
	temp.append(0)
grad= []                                       
for i in range (0, feat):                                       
	grad.append(0)
gradb= 0
ada= []                                       
for i in range (0, feat):                                       
	ada.append(0)

idx= 0###############################抓data
take=[0,2,3,4,5,6,9,10,11,14,15,16,17,18,19,20,21,24,25,26,27,28,29,31,33,34,35,36,37,38,40,42,43,44,45,46,49,51,52,54,56,57,58,62,65,70,77,85,86,88,90,93,96,101,102,105]
reader= csv.reader(open(sys.argv[3], 'r'))
idx= -1
for row in reader:  
	if idx==-1:
		idx=0
		continue                     
	for j in range (0, 106): 
		data[idx][j]=float(row[j])
	idx= idx+1
reader= csv.reader(open(sys.argv[4], 'r'))
idx= 0
for row in reader:  
	y[idx]=int(row[0])
	idx= idx+1
	if idx==id1:
		break
for i in range (0, id1):    ###train      
	for j in range (0, feat):
		twod[i][j]=data[i][j] 
a=np.array(twod)
std=np.std(a,axis=0)
print(std)
mean=np.mean(a,axis=0)
print(mean)
for i in range (0, id1):        
	for j in range (0, feat):
		if(std[j]==0):
			std[j]=1
		twod[i][j]=(twod[i][j]-mean[j])/std[j]
twod=np.array(twod, dtype=np.float64)
w=np.array(w, dtype=np.float64)
y=np.array(y)
print(twod)
print("\n\n")
print(y)

for i in range (0, id1):     ########validate
	for j in range (0, feat):
		valid[i][j]=data[i][j]
a=np.array(valid)
std=np.std(a,axis=0)
print(std)
mean=np.mean(a,axis=0)
print(mean)
for i in range (0, id1):        
	for j in range (0, feat):
		if(std[j]==0):
			std[j]=1
		valid[i][j]=(valid[i][j]-mean[j])/std[j]

print("--------------------------------------------------------")

def foo():
	
	z=np.dot(twod,np.transpose(w))+b
	f=1/( 1 + 2.718281**(-z) )
	output=0
	for i in range (0,id1):
		output=output-(y[i]-f[i])
	return output

def foox():
	z=np.dot(twod,np.transpose(w))+b
	f=1/( 1 + 2.718281**(-z) )
	output=0
	for i in range (0,id1):
		for j in range (0,feat):
			temp[j]=temp[j]-twod[i][j]*(y[i]-f[i])
	for j in range (0,feat):
			temp[j]=temp[j]+2*lamb*w[j]
best=0
def loss():
	global best
	right= 0
	pr=0
	p=0
	nr=0
	n=0
	#for i in range (10000+x*3000,13000+x*3000):
	z=np.dot(twod,np.transpose(w))+b
	for i in range (0,id1):
		if z[i]>0 :
			p=p+1
			if y[i]==1:
				right= right+1
				pr=pr+1
		if z[i]<0 :
			n=n+1
			if y[i]==0:
				right= right+1
				nr=nr+1
	rate=right/id1
	print("rate=",rate,p,pr,n,nr)
	if rate>best:
		best=rate
	print("best=",best,"\n")
	return rate
print("----------------training start----------------")
while 1:#72825
	e= 1
	for i in range (0,feat): 
		temp[i]=0
	foox()
	print(temp,end=",")
	for i in range (0,feat): 
		grad[i]=grad[i]+temp[i]**2 
	tempb=foo()
	print(tempb)
	gradb=gradb+tempb**2
	for i in range (0,feat): 
		if grad[i]==0:
			ada[i]= 1
		else:
			ada[i]= sqrt(grad[i])
	adab= sqrt(gradb)
	for i in range (0,feat): #gradient decent
		w[i]=w[i]-e/ada[i]*temp[i]
		print(w[i],end=",")
	b=b-e/adab*tempb
	print(b)
	if loss()==0.8547058823529412:
		break


print("----------------training end----------------")








print("-----------------------------output---------------------------")
twod = []       ########### arrays                             
for i in range (0, 16281):                  
	new = []                 
	for j in range (0, feat):    
		new.append(0)     
	twod.append(new)
result = []       ########### arrays                             
for i in range (0, 16281):                      
	result.append(0)    

idx= 0################抓data
reader= csv.reader(open(sys.argv[5], 'r'))
idx= -1
for row in reader:  
	if idx==-1:
		idx=0
		continue                     
	for j in range (0, 106): 
		data[idx][j]=row[j]
	idx= idx+1

for i in range (0, 16281):                               
	for j in range (0, feat):
		twod[i][j]=float(data[i][j])
a=np.array(twod)
std=np.std(a,axis=0)
print(std)
mean=np.mean(a,axis=0)
print(mean)
for i in range (0, 16281):        
	for j in range (0, feat):
		if(std[j]==0):
			std[j]=1
		twod[i][j]=(twod[i][j]-mean[j])/std[j]
right=0#####################算算
for i in range (0, 16281):
	temp= 0
	for j in range (0,feat):
		temp= temp+w[j]*twod[i][j]
	z= temp+b

	if z>0 :
		result[i]=1
	if z<0 :
		result[i]=0



with open(sys.argv[6],"w", newline='') as f:###########輸出
	w = csv.writer(f)
	w.writerow(['id','label'])
	for i in range (0, 16281):  
		a=str(i+1)
		new = []                 
		new.append(a) 
		new.append(str(result[i]))
		w.writerow(new)
	#print(data)
	f.close()

#print(test)