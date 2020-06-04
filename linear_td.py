# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:26:57 2020

@author: qinzhen
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from env import *

def linear_td(theta, lambda_, epsilon=0.05, gamma=1, alpha=0.01):
    #初始化
    E = np.zeros_like(theta)
    flag = False
    s = init()
    key = tuple(s)
    a = greedy(theta, key, epsilon)
    env = Easy21()
    while not flag:
        key = tuple(s)
        #执行动作
        s, r, flag = env.step(s, a)
        #特征
        f = get_feature(key, a)
        #选择下一步动作
        a1 = greedy(theta, key, epsilon)
        key1 = tuple(s)
        #特征
        f1 = get_feature(key1, a1)
        #计算delta
        delta = r - np.dot(theta, f)
        #判断是否终止
        if not flag:
            delta += gamma * np.dot(theta, f1)
        E = gamma * lambda_ * E + f
        theta += alpha * delta * E
        a = a1
        
def computeQ(theta):
    Q = dict()
    X = np.arange(1, 11)
    Y = np.arange(1, 22)
    A = [0, 1]
    for x in X:
        for y in Y:
            s = (x, y)
            Q[s] = np.array([0.0, 0.0])
            for a in A:
                f = get_feature(s, a)
                Q[s][a] = np.dot(theta, f)
    return Q

with open("mcQ.pickle", "rb") as file:
    Q1 = pickle.load(file)
    
Error = []
Lambda = np.linspace(0, 1, 11)
K = 10000

for lambda_ in Lambda:
    theta = np.zeros(36)
    for i in range(K):
        linear_td(theta, lambda_)
    Q = computeQ(theta)
    error = error_square(Q, Q1)
    Error.append(error)
    if lambda_ == 0 or lambda_ == 1:
        filename = "linear_td" + str(int(lambda_)) + ".pickle"
        with open(filename, "wb") as file:
            pickle.dump(Q, file)
            
plt.plot(Lambda, Error)
plt.show()