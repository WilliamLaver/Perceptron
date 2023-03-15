#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 20:25:30 2023

@author: william

Perceptron executes successfully on existing test cases, hyperplane plotting
function has a bug, cannot perform random test cases without visual aid (why? 
can implement quality analysis function?)

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(1, 1, 1, projection='3d')
sns.set_palette('bright')

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

def mkunitary(a, n):
    unitary = np.array([[0 for i in range(n)] for i in range(n)])
    for i in range(n):
        for j in range(n):
            if i == j:
                unitary[i][j] = a

    return unitary

def func_z(x, y, a, b, c, d):
    return (-b / a) * x + (-c / a) * y + d

def mkplane(th, th0, x_min, x_max):
    
    a = th[0][0]
    b = th[1][0]
    c = th[2][0]

    x = np.array([x_min, (x_max-x_min)/2, x_max])
    y = np.array([x_min, (x_max-x_min)/2, x_max])
    z = np.array([x_min, (x_max-x_min)/2, x_max])
    
    if a == 0:
        d = th0 / th[0][2]
        x1, z1 = np.meshgrid(x, z)
        y1 = func_z(x1, z1, b, a, c, d)
        ax.plot_wireframe(x1, z1, y1)
        
    elif b == 0:
        d = th0 / th[0][2]
        x1, y1 = np.meshgrid(x, y)
        z1 = func_z(x1, y1, b, a, c, d)
        ax.plot_wireframe(x1, y1, z1)
        
    else:
        d = th0 / th[0][0]
        y1, z1 = np.meshgrid(y, z)
        x1 = func_z(y1, z1, a, b, c, d)
        ax.plot_wireframe(y1, z1, x1)
        
    return True


data = np.array([(0,0,0),(0,0,1),(0,1,0),
                 (0,1,1),(1,0,0),(1,0,1),
                 (1,1,0),(1,1,1)]).transpose()

labels = np.array([-1, -1, -1, -1, -1, -1, -1, 1])

colours = colour(labels)

ax.scatter(data[0, :], data[1, :], data[2, :], c=colours)

th = np.array([[0], [0], [0]])
th0 = 0
errs = []
ceiling = 1000

exists_conflict = True

while (exists_conflict):
    
    exists_conflict = False
    
    for i in range(len(data[0,:])):
        dist = (th.T @ data[:, i] + th0)
        if ((th.T @ data[:, i] + th0) * labels[i]) <= 0:

            exists_conflict = True
            errs.append((i, th.T))
            
            y = mkunitary(labels[i], len(data[:, 0]))
            x = data[:, i]
            
            th = np.add(th.T, y @ x).reshape(3, 1)
            th0 += labels[i]
            # process_trial(data, th, th0)
    
    if len(errs) > ceiling:
        exists_conflict = False
        print("Too many tries...\n")

print("theta: ", th, "\ntheta_0: ", th0, "\nErrors: ", len(errs))
mkplane(th, th0, min(data[0, :]), max(data[0, :]))
plt.show()