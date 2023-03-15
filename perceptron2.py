#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 20:27:21 2023

@author: william

"""

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
colours = np.array([])


def mkunitary(a, n):
    unitary = np.array([[0 for i in range(n)] for i in range(n)])
    for i in range(n):
        for j in range(n):
            if i == j:
                unitary[i][j] = a

    return unitary


def mkline(th, th0, x_min, x_max):
    
    if th[0][0] == 0:
        line = np.array([list(range(-5, 6, 2)), [th[1][0]] * 6])
        print(line)
    elif th[1][0] == 0:
        line = np.array([[th[0][0]] * 6, list(range(-5, 6, 2))])
    else:
        m = - th[0][0] / th[1][0]
        b = - th0 / th[1][0]
        
        line = np.array([[x_min, m * x_min + b], [0, b], [1, b+m],
                     [x_max, m * x_max + b]]).transpose()
    
    return line


# create an array of colour values to visualize the labels
def colour(labels):
    n = len(labels)
    colours = np.array(['#1821b5' for i in range(n)])
    for i in range(len(labels)):
        if labels[i] <= 0:
            colours[i] = '#1821b5'
        else:
            colours[i] = '#d05703'
            
    return colours


# this function produces n randomly sampled data points and
# assigns a label to each as an added dimension, can isolate
# data using D[0:-1, :], domain: 0 -> 10
def generator(dim, n):
    data = np.random.rand(dim, n) @ mkunitary(10, n)
    labels = assignlabels(data)
    D = np.concatenate((data,labels))
    return D


# this function defines a correlation in the dataset
def assignlabels(data):
    
    n = len(data[0])
    dx = max(data[0, :]) - min(data[0, :])
    dy = max(data[1, :]) - min(data[1, :])
    labels = np.array([[0 for i in range(n)]])
    
    for i in range(n):
        
        #if data[0][i] >= dx*0.3 and data[1][i] >= dy*0.6:
        if data[1][i] >= (0.60*data[0][i] + 0.1):
            labels[0][i] = 1
        else:
            labels[0][i] = -1
            
    return labels


def process_trial(data, th, th0):
    best_fit = mkline(th, th0, min(data[0, :]), max(data[0, :]))
    ax.scatter(data[0, :], data[1, :], c=colours)
    ax.plot(best_fit[0, :], best_fit[1, :])


def process_winner(data, th, th0):
    best_fit = mkline(th, th0, min(data[0, :]), max(data[0, :]))
    ax.scatter(data[0, :], data[1, :], c=colours)
    ax.plot(best_fit[0, :], best_fit[1, :], c='#f70455', lw=2)


""" DATASETS:
_________________________________________________________________________
data = np.array([[-3, 2], [-1, 1], [-1, -1], [2, 2],[1, -1]]).transpose()
labels = np.array([1, -1, -1, -1, -1])

data = np.array([[1,  2], [1,  3], [2,  1], [1, -1], [2, -1]]).transpose()
labels = np.array([-1, -1, 1, 1, 1])

data = np.array([[1, -1], [0, 1], [-1.5, -1]]).transpose()
labels = np.array([1, -1, 1])

data = np.array([[1, -1], [0, 1], [-10, -1]]).transpose()
labels = np.array([1, -1, 1])

D = generator(2,50)
data = np.array(D[0:-1, :])
labels = np.array(D[-1, :])
"""
D = generator(2,50)
data = np.array(D[0:-1, :])
labels = np.array(D[-1, :])

colours = colour(labels)

th = np.array([[0], [0]])
th0 = 0
errs = []

exists_conflict = True

while (exists_conflict):
    
    exists_conflict = False
    
    for i in range(len(data[0, :])):
        dist = (th.T @ data[:, i] + th0)
        if (dist * labels[i]) <= 0:
            
            exists_conflict = True
            errs.append((i, th.T))
            
            y = mkunitary(labels[i], len(data[:, 0]))
            x = data[:, i]
            
            th = np.add(th.T, y @ x).reshape(2, 1)
            th0 += labels[i]
            # process_trial(data, th, th0)

print("theta: ", th, "\ntheta_0: ", th0, "\nErrors: ", len(errs))

process_winner(data, th, th0)
