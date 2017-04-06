# -*- coding: utf-8 -*- 
import csv
from numpy import *
import pandas as pd
from numpy import genfromtxt
import numpy as np
import sys

size=106
test = []       ########### original data                            
for i in range (0, 16281):                
	new = []                 
	for j in range (0, size):    
		new.append(0)     
	test.append(new)
result = []       ########### original data                            
for i in range (0, 32561):                
	new = []                 
	for j in range (0, size):    
		new.append(0)     
	result.append(new)
data = []       ########### original data                            
for i in range (0, 32561):                
	new = []                 
	for j in range (0, size):    
		new.append('0')     
	data.append(new)
y = []                                       
for i in range (0, 32561):                                       
	y.append(0)  
pos = []       ########### arrays                             
for i in range (0, 7841):                  
	new = []                 
	for j in range (0, size):    
		new.append(0)     
	pos.append(new)
nag = []       ########### arrays                             
for i in range (0, 24720):                  
	new = []                 
	for j in range (0, size):    
		new.append(0)     
	nag.append(new)

idx= 0###############################抓data
reader= csv.reader(open(sys.argv[3], 'r'))
idx= -1
for row in reader:  
	if idx==-1:
		idx=0
		continue                     
	for j in range (0, size): 
		data[idx][j]=float(row[j])
	idx= idx+1
reader= csv.reader(open(sys.argv[4], 'r'))
idx= 0
for row in reader:  
	y[idx]=int(row[0])
	idx= idx+1
print("ok")

idp=0
idn=0
for i in range (0, 32561):  
	if(y[i]==1):      
		for j in range (0, size):
			pos[idp][j]=data[i][j]
		idp=idp+1   
	elif y[i]==0:
		for j in range (0, size):
			nag[idn][j]=data[i][j]
		idn=idn+1
print(idp,idn)
###############################算算
#np.set_printoptions(threshold=np.nan)
mpos=np.matrix(pos)
mnag=np.matrix(nag)
pm=np.mean(pos,axis=0)
nm=np.mean(nag,axis=0)
psig=0
for i in range (0,idp):
	psig=psig+dot(np.transpose((mpos[i]-pm)),(mpos[i]-pm))
psig=psig/idp
nsig=0
for i in range (0,idn):
	nsig=nsig+dot(np.transpose((mnag[i]-nm)),(mnag[i]-nm))
nsig=nsig/idn
sig=(idp*psig+idn*nsig)/(idp+idn)
def P(x):
	a=np.array(dot(dot(x-pm,linalg.inv(sig)),np.transpose(x-pm))  / (-2) )
	pe=2.718281**a[0][0]
	b=np.array(dot(dot(x-nm,linalg.inv(sig)),np.transpose(x-nm))  / (-2) )
	ne=2.718281**b[0][0]
	c1=idp/(idp+idn)
	c2=idn/(idp+idn)
	#print(1/( 1 + (ne*c2)/(pe*c1) ),pe,ne)
	return 1/( 1 + (ne*c2)/(pe*c1) )
count=0
upup=0
down=0
"""
for i in range (0,32561):  
	output=P(data[i])
	print(i+1,"=",output)
	if output>0.5:
		upup=upup+1
		result[i]=1
	else:
		result[i]=0
		down=down+1
	if result[i]==y[i]:
		count=count+1
print(count/32561,32561,"=",upup,down)
"""

reader= csv.reader(open(sys.argv[5], 'r'))##抓抓
idx= -1
for row in reader:  
	if idx==-1:
		idx=0
		continue                     
	for j in range (0, size): 
		test[idx][j]=float(row[j])
	idx= idx+1

for i in range (0,16281):  ##算算
	output=P(test[i])
	print(i+1,"=",output)
	if output>0.5:
		upup=upup+1
		result[i]=1
	else:
		result[i]=0
		down=down+1
print(down/(upup+down),upup,down)

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
	