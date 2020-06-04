# -*- coding: utf-8 -*-
"""
Created on Fri May  8 22:52:02 2020

@author: qinzhen
"""

from env import *
import numpy as np
import pickle

gamma = 1
N0 = 100
Q = get_dict()
N = get_dict()
K = 1000000

def mc(Q, N, N0=100, gamma=1):
    flag = False
    s = init()
    env = Easy21()
    #轨迹
    trace = []
    while not flag:
        key = tuple(s)
        a = epsilon_greedy(Q, N, key, N0)
        s, r, flag = env.step(s, a)
        trace.append((key, a, r))
        #更新
        N[key][a] += 1
    
    G = 0
    n = len(trace)
    for i in range(n-1, -1, -1):
        s, a, r = trace[i]
        G = gamma * G + r
        alpha = 1 / N[s][a]
        Q[s][a] += alpha * (G - Q[s][a])
        
for i in range(K):
    mc(Q, N)
    
#保存
with open("mcQ.pickle", "wb") as file:
    pickle.dump(Q, file)