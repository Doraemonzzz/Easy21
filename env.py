# -*- coding: utf-8 -*-
"""
Created on Fri May  8 22:28:55 2020

@author: qinzhen
"""

import numpy as np

dealer = [[1, 4], [4, 7], [7, 10]]
player = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]
action = [0, 1]

def update(s, i):
    """
    Parameters
    ----------
    s : state array
    i : index
    
    Returns
    -------
    None.
    """
    #生成概率
    p = np.random.rand()
    v = np.random.randint(1, 11)
    if p < 1 / 3:
        s[i] -= v
    else:
        s[i] += v
        
def init():
    return np.random.randint(1, 11, 2)

def judge(s, i):
    #判断是否结束
    if s[i] > 21 or s[i] < 1:
        return True
    return False

def get_action():
    return np.random.randint(2)

class Easy21():
    def step(self, s, a):
        """
        Parameters
        ----------
        s : state array, dealer s[0], player s[1]
        a : action, hit: 0; stick: 1

        Returns
        -------
        next state s'
        reward r
        terminal flag
        """
        #判断是否结束
        flag = False
        r = 0
        if a == 0:
            update(s, 1)
            if judge(s, 1):
                r = -1
                flag = True
        else:
            flag = True
            while s[0] < 17 and s[0] > 0:
                update(s, 0)
            if judge(s, 0):
                r = 1
            else:
                if s[0] < s[1]:
                    r = 1
                elif s[0] == s[1]:
                    r = 0
                else:
                    r = -1
        
        return s, r, flag
    
def epsilon_greedy(Q, N, key, N0):
    n = N[key].sum()
    epsilon = N0 / (N0 + n)
    p = np.random.rand()
    if p >= epsilon:
        a = np.argmax(Q[key])
    else:
        a = get_action()
        
    return a

def get_feature(s, a):
    feature = []
    for d in dealer:
        for p in player:
            for act in action:
                if s[0] >= d[0] and s[0] <= d[1] and s[1] >= p[0] and s[1] <= p[1] and a == act:
                    feature.append(1)
                else:
                    feature.append(0)
                    
    return np.array(feature)

def greedy(theta, s, epsilon):
    p = np.random.rand()
    f1 = get_feature(s, 0)
    f2 = get_feature(s, 1)
    v = [np.dot(theta, f1), np.dot(theta, f2)]
    if p >= epsilon:
        a = np.argmax(v)
    else:
        a = get_action()
        
    return a

def error_square(Q1, Q2):
    error = 0
    for key in Q1:
        error += (np.max(Q1[key]) - np.max(Q2[key])) ** 2;
    
    return error

def get_dict():
    X = np.arange(1, 11)
    Y = np.arange(1, 22)
    res = dict()
    for x in X:
        for y in Y:
            res[(x, y)] = np.array([0.0, 0.0])
    
    return res