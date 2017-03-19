# -*- coding: utf-8 -*- 
import csv
from numpy import *
import pandas as pd
import os
import sys

def rain(k):
	if k is 'NR':
		return 0
	else:
		return 1

take= 8
feat= 8

twod = []       ########### arrays                             
for i in range (0, 800):                  
	new = []                 
	for j in range (0, feat):    
		new.append(0)     
	twod.append(new)
pm = []                                       
for i in range (0, 800):                                       
	pm.append(0)  
b= 0
w= []                                       
for i in range (0, feat):                                       
	w.append(0)
temp= []                                       
for i in range (0, feat):                                       
	temp.append(1)
grad= []                                       
for i in range (0, feat):                                       
	grad.append(0)
gradb= 0
ada= []                                       
for i in range (0, feat):                                       
	ada.append(1)

idx= 0################抓data
id1= 0
df=pd.read_csv(sys.argv[1], sep=',',header=None,encoding="big5")
for row in range(0,240):
	y=1+row*18
	for col in range(3,27):
		if(idx == take):
			pm[id1]= float(df[col][y+9])
		else:
			twod[id1][idx]=float(df[col][y+9])
			if idx == 9999:
				twod[id1][9]=float(df[col][y+0])
				twod[id1][10]=float(df[col][y+1])
				twod[id1][11]=float(df[col][y+2])
				twod[id1][12]=float(df[col][y+3])
				twod[id1][13]=float(df[col][y+4])
				twod[id1][14]=float(df[col][y+5])
				twod[id1][15]=float(df[col][y+6])
				twod[id1][16]=float(df[col][y+7])
				twod[id1][17]=float(df[col][y+8])
				twod[id1][18]=float(df[col][y+11])
				twod[id1][19]=float(df[col][y+12])
				twod[id1][20]=float(df[col][y+13])
				twod[id1][21]=float(df[col][y+14])
				twod[id1][22]=float(df[col][y+15])
				twod[id1][23]=float(df[col][y+16])
				twod[id1][24]=float(df[col][y+17])

		idx= idx+1
		if idx == take+1:
			id1= id1+1
			idx= 0

print(id1)
print(twod)
print("\n\n")
print(pm)
print("--------------------------------------------------------")
id1=320

def foo():
	output= 0
	for i in range (0,id1):
		temp= 0
		for j in range (0,feat):
			temp= temp+w[j]*twod[i][j]
		output= output-2*(pm[i]-(b+temp))
	return output
def foox(x):
	output= 0
	for i in range (0,id1):
		temp= 0
		for j in range (0,feat):
			temp= temp+w[j]*twod[i][j]
		output= output-2*twod[i][x]*(pm[i]-(b+temp))
	return output
min=99999
def loss():
	global min
	all= 0
	output= 0
	for i in range (0,id1):
		temp= 0
		for j in range (0,feat):
			temp= temp+w[j]*twod[i][j]
		output= output+(pm[i]-(b+temp))**2
		all= all+(pm[i]-(b+temp))**2
	diff= sqrt(all/id1)
	if diff<min:
		min= diff
	print(diff," min=",min)
	print("LOSS=",output)
	return diff



print("----------------training start----------------")
while loss()>5.58224702012:#72825
	e= 1
	for i in range (0,feat): #gradient
		temp[i]= foox(i)
		print(temp[i],end=",")
		grad[i]=grad[i]+temp[i]**2 
	tempb=foo()
	print(tempb)
	gradb=gradb+tempb**2
	for i in range (0,feat): 
		ada[i]= sqrt(grad[i])
	adab= sqrt(gradb)
	for i in range (0,feat): #gradient decent
		w[i]=w[i]-e/ada[i]*temp[i]
		print(w[i],end=",")
	b=b-e/adab*tempb
	print(b)

print("----------------training end----------------")

########################抓test
test= []
for i in range (0, 240):                  
	new = []                 
	for j in range (0, 19):    
		new.append(0)     
	test.append(new)
idx= 0
id1= 0
df=pd.read_csv(sys.argv[2], sep=',',header=None,encoding="big5")
for row in range(0,240):
	y= row*18
	x= 2

	test[id1][0]=float(df[x+1][y+9])
	test[id1][1]=float(df[x+2][y+9])
	test[id1][2]=float(df[x+3][y+9])
	test[id1][3]=float(df[x+4][y+9])
	test[id1][4]=float(df[x+5][y+9])
	test[id1][5]=float(df[x+6][y+9])
	test[id1][6]=float(df[x+7][y+9])
	test[id1][7]=float(df[x+8][y+9])

	id1= id1+1

#print(test)
result = []  #########################算結果                                     
for i in range (0, 240):                                       
	result.append(0)  

count=0
total=0
for j in range (0, 240):
	for i in range (0, feat):                                       
		result[j]= result[j]+w[i]*test[j][i]
	result[j]= result[j]+b
	if (pm[j]-result[j])**2>100:
		count=count+1
	total=total+abs(pm[j]-result[j])

directory = os.path.dirname(sys.argv[3])#####################輸出
if not os.path.exists(directory):
	if '/' in sys.argv[3]:
		os.makedirs(directory)
a= []
s= []
for i in range (0, 240):  
		a.append("id_"+str(i))              
		s.append(result[i]) 

df = pd.DataFrame({'id':a, 'value':s})
df.to_csv(sys.argv[3], index=False)

print("w=",w)
print("b=",b)
print("The output is in",directory)