# -*- coding: utf-8 -*-
"""
Created on Sat May  9 01:07:06 2020

@author: qinzhen
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from env import *

def td(Q, N, lambda_, N0=100, gamma=1):
    flag = False
    s = init()
    key = tuple(s)
    a = epsilon_greedy(Q, N, key, N0)
    env = Easy21()
    E = get_dict()
    while not flag:
        key = tuple(s)
        #执行动作
        s, r, flag = env.step(s, a)
        #选择下一步动作
        a1 = epsilon_greedy(Q, N, key, N0)
        key1 = tuple(s)
        #更新计数
        N[key][a] += 1
        #更新E
        E[key][a] += 1
        #计算delta
        delta = r - Q[key][a]
        #判断是否终止
        if not flag:
            delta += gamma * Q[key1][a1]
        #更新Q
        alpha = 1 / N[key][a]
        for key in Q:
            Q[key] += alpha * delta * E[key]
            E[key] *= gamma * lambda_
        
        #赋值
        a = a1
        
with open("mcQ.pickle", "rb") as file:
    Q1 = pickle.load(file)

Lambda = np.linspace(0, 1, 11)
N0 = 100
K = 10000
Error = []

for lambda_ in Lambda:
    Q = get_dict()
    N = get_dict()
    for i in range(K):
        td(Q, N, lambda_)
    error = error_square(Q, Q1)
    Error.append(error)
    if lambda_ == 0 or lambda_ == 1:
        filename = "td" + str(int(lambda_)) + ".pickle"
        with open(filename, "wb") as file:
            pickle.dump(Q, file)
         
plt.plot(Lambda, Error)
plt.show()