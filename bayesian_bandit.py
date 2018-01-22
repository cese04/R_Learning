# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 00:09:51 2018

@author: CarlosEmiliano
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

NUM_TRIALS = 200
BANDIT_PROBABILITIES = [0.2, 0.5]

class Bandit:
    def __init__(self, p):
        self.p = p
        self.a = 1
        self.b = 1
        
    def pull(self):
        return np.random.random() < self.p
    
    def sample(self):
        return np.random.beta(self.a, self.b)
    
    def update(self, x):
        self.a += x
        self.b += 1-x
        
def plot(bandits, trial):
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label='real p: %.4f' %b.p)
    plt.title("Bandit distribution after %s trials" %trial)
    plt.legend()
    plt.show()
    
    
def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 19999]
    for i in range(NUM_TRIALS):
        bestb = None
        maxsample = -1
        allsamples = []
        for b in bandits:
            sample = b.sample()
            allsamples.append("%.4f" %sample)
            if sample > maxsample:
                maxsample = sample
                bestb = b
        if i in sample_points:
            print("Current samples: %s" %allsamples)
            plot(bandits, i)
            
        x = bestb.pull()
        bestb.update(x)
    return bandits
        
if __name__ == '__main__':
    bandits = experiment()
    s = 0
    for i in range(len(bandits)):
        s += bandits[i].a
    print(s)
    X = np.linspace(0,1,200)
    y = beta.pdf(X, bandits[0].a, bandits[0].b)
    y2 = beta.pdf(X, bandits[1].a, bandits[1].b)
    Y = np.matmul(np.transpose([y]),[y2])
        
    plt.imshow(Y)


