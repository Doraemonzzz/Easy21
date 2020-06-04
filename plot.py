# -*- coding: utf-8 -*-
"""
Created on Sat May  9 00:34:28 2020

@author: qinzhen
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot(filename, title):
    with open(filename, "rb") as file:
        Q = pickle.load(file)
        
    X = np.arange(1, 11)
    Y = np.arange(1, 22)
    Z = np.zeros((21, 10))
    
    for key in Q:
        x = key[0]
        y = key[1]
        Z[y - 1][x - 1] = np.max(Q[key])
        
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax.set_xlabel('Dealer showing')
    ax.set_ylabel('Play sum')
    ax.set_zlabel('value')
    plt.title(title)
    plt.show()
    
mc = "mcQ.pickle"
td0 = "td0.pickle"
td1 = "td1.pickle"
linear_td0 = "linear_td0.pickle"
linear_td1 = "linear_td1.pickle"

plot(mc, "mc")
plot(td0, "td0")
plot(td1, "td1")
plot(linear_td0, "linear_td0")
plot(linear_td1, "linear_td1")
