# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 19:45:06 2017

@author: CarlosEmiliano
"""

import numpy as np
#from scipy import stats
import matplotlib.pyplot as plt


#Nm = 10000
#a = np.random.randn(Nm) + 0.3
#b = np.random.randn(Nm) + 0.37

#print(a.mean())
#print(b.mean())

def generate(case):
    if case==0:
        mu = 0.5
    elif case==1:
        mu = 0.6
    click = 1 if (np.random.random() < mu) else 0
    #click2 = 1 if (np.random.random() < self.p2) else 0
    return click

N = 0
Nj = np.zeros(2)
epsilon = np.zeros(2)
mu = np.ones(2)
acc = np.zeros(2)

itera = 1000000
for i in range(itera):
    N = i + 1
    if i<10 :
        m = round(np.random.random())
        acc[m] += generate(m)
        Nj[m] += 1
        mu = acc/Nj
        epsilon = np.sqrt((2*np.log(N))/Nj)
        
    else:
        m = np.argmax(mu+epsilon)
        #m = round(np.random.random())
        acc[m] += generate(m)
        Nj[m] += 1
        mu = acc/Nj
        epsilon = np.sqrt((2*np.log(N))/Nj)
        
    #acc[0] += generate(0)
    #acc[1] += generate(1)
    #mu

print("Veces jugadas: " , Nj) 
print("Veces ganadas: ", acc)
print("Media: ", mu)

print("Total: ", acc.sum())



