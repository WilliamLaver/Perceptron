#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:51:04 2023

@author: william
"""
import numpy as np
import matplotlib.pyplot as plt

# GLOBALS
fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(111)


def mkcircle(center, r):

    h = center[0][0]
    k = center[1][0]

    x = np.array([r * np.cos(th) + h for th in np.linspace(0, 2 * np.pi, 100)])
    y = np.array([r * np.sin(th) + k for th in np.linspace(0, 2 * np.pi, 100)])

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.plot(x, y)
    return True  # np.array(x, y)


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


# this function defines a correlation in the dataset
def assignlabels(data):

    n = len(data[0])
    dx = max(data[0, :]) - min(data[0, :])
    dy = max(data[1, :]) - min(data[1, :])
    labels = np.array([[0 for i in range(n)]])

    for i in range(n):

        # if data[0][i] >= dx*0.3 and data[1][i] >= dy*0.6:
        # if data[1][i] >= (0.60*data[0][i] + 0.1):
        if (data[0][i]**2 + data[1][i]**2)**0.5 <= 6:
            labels[0][i] = 1
        else:
            labels[0][i] = -1

    return labels


# Returns the unitary matrix with specified weight and dimension
def mkunitary(a, n):
    unitary = np.array([[0 for i in range(n)] for i in range(n)])
    for i in range(n):
        for j in range(n):
            if i == j:
                unitary[i][j] = a

    return unitary


# Generator function
# this function produces n randomly sampled data points and
# assigns a label to each as an added dimension, can isolate
# data using D[0:-1, :], domain: 0 -> 10
def generator(dim, n):
    data = np.random.rand(dim, n) @ mkunitary(10, n)
    labels = assignlabels(data)
    D = np.concatenate((data, labels))
    return D


# Kernel functions
def K(th, x):
    return th.T @ x


def K1(th, x):
    return (th.T @ x)**2


def K2(th, x, th0):

    h = np.array(th[0][0])
    k = np.array(th[1][0])
    th_trn = np.array([[k**2 + h**2 - th0**2], [-2 * h], [-2 * k], [1], [1]])
    x_trn = np.array([1, x[0], x[1], x[0]**2, x[1]**2])

    res = th_trn.T @ x_trn
    return res[0]

""" DATASETS:
    
D = generator(2, 50)
data = np.array(D[0:-1, :])
labels = np.array(D[-1, :])

data = np.array([[-1, 1], [1, -1], [1, 1], [2, 2]]).transpose()
labels = np.array([1, 1, -1, -1])

"""
D = generator(2, 50)
data = np.array(D[0:-1, :])
labels = np.array(D[-1, :])

colours = colour(labels)

# in this case theta is the normal to the circle
th = np.array([[0], [0]])
th0 = 0

center = np.array([[0], [0]])
r = 1

errNum = 0
exists_conflict = True

while (exists_conflict):

    exists_conflict = False
    
    for i in range(len(data[0, :])):
        
        dist = K2(th, data[:, i], th0)

        if dist * labels[i] <= 0:
            
            errNum += 1
            y = mkunitary(labels[i], len(data[:, 0]))
            x = data[:, i]
            
            th = np.add(th.T, y @ x).reshape(2, 1)
            th0 += errNum

print("th: ", th, "th_0: ", th0)
mkcircle(th, th0)
#   mkcircle(np.array([[0], [0]]), 6)
# mkcircle([[-3], [-3]], 5, 0, 10)
ax.scatter(data[0, :], data[1, :], c=colours)
