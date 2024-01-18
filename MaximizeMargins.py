#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:36:36 2024

@author: william
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt


class MarginMax(object):

    def __init__(self, N, dim, D=None, alg='perceptron'):
        self.N = N
        self.dim = dim 
        self.alg = alg
        self.mat = None
        self.fig = plt.figure(figsize=plt.figaspect(1))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.N = N

        self.main(D)
        return None

# ----------------------------- UTILITIES ------------------------------------

    # returns a signed distance to the point x from plane th
    def dist(self, x, H):
        if not H.any():
            return 0
        th0 = H[-1]
        th = H[:-1]
        return (th.T @ x + th0) / (th.T @ th)**.5

    def h_func(self, x, y, a, b, c, d):
        if c != 0:
            out = -1 * (a * x + b * y + d) / c
        else:
            out = 0
        return out

# -------------------------- GRAPHING FUNCTIONS------------------------

    def visualize(self, D, H):
        plt.clf()
        self.fig = plt.figure(figsize=plt.figaspect(1))
        if self.dim == 3:
            self.visualize3D(D, H)
        elif self.dim == 2:
            self.visualize2D(D, H)
        plt.show()
        return None

    def visualize3D(self, D, H):
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.plotH_3D(H)
        self.plotD_3D(D)
        return None

    def plotH_3D(self, H):
        if H.any():
            xvals = np.linspace(0, 1, self.N)
            yvals = np.linspace(0, 1, self.N)
            X, Y = np.meshgrid(xvals, yvals)
            Z = self.h_func(X, Y, H[0][0], H[1][0], H[2][0], H[3][0])
            self.ax.plot_wireframe(X, Y, Z)
        return None

    def plotD_3D(self, D):
        L = D[-1]
        C = self.colour(L)
        D = D[:-1]
        self.ax.scatter(D[0, :], D[1, :], D[2, :], c=C)
        return None

    def visualize2D(self, D, H):
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        m = - H[:-1][0] / H[:-1][1]
        b = - H[-1] / H[:-1][1]
        x = np.arange(0, 1, 0.01)
        y = m * x + b
        self.ax.plot(x, y)
        C = self.colour(D[-1])
        self.ax.scatter(D[:-1][0], D[:-1][1], c=C)
        return None

    # create an array of colour values to visualize the labels
    def colour(self, L, pal=1):
        if pal == 1:
            c1 = '#d94a02'
            c2 = '#01912a'
        else:
            c1 = '#0db902'
            c2 = '#990170'
        colours = np.array([c1 for i in range(len(L))])
        for i in range(len(L)):
            if L[i] <= 0:
                colours[i] = c1
            else:
                colours[i] = c2
        return colours
        

#-------------------------------DATA GENERATION-------------------------------

    def separate(self, D, q=.03):
        l = np.array([0 for i in range(len(D[0]))])
        h = np.array([[np.random.rand()] for i in range(self.dim + 1)])
        h[-1] = - np.array([[0.5 for i in range(self.dim)]]) @ h[:-1]
        for i in range(len(D[0])):
            if self.dist(D[:, i], h) >= 0:
                l[i] = 1
                D[:, i] = D[:, i] + [q for i in range(self.dim)]
            else:
                l[i] = -1
                D[:, i] = D[:, i] - [q for i in range(self.dim)]
        return np.vstack((D, l))

    # generate labelled data points in 3 dimensions
    def gen_data(self):
        D = self.gen_lin_sep()
        D = self.separate(D)
        return D

    def gen_lin_sep(self):
        D = np.array([np.random.rand(self.N) for i in range(self.dim)])
        return D

# -----------------------LEARNING ALGORITHMS----------------------------------
   
    def train(self, D):
        H, gamma, errs = self.perceptron(D)
        print("gamma: " + str(gamma) + " \ errors: " + str(errs))
        return H

    def perceptron(self, D):
        H = np.array([[0.0] for i in range(self.dim + 1)])
        R = D[:-1]
        L = D[-1]
        gamma = 1
        errs = 0
        error_exists = True
        while (error_exists):
            error_exists = False
            for i in range(len(R[0])):
                d = self.dist(R[:, i], H)
                if d * L[i] <= 0:
                    error_exists = True
                    errs += 1
                    H[:-1] = np.add(H[:-1].T, L[i] * R[:, i]).T
                    H[-1] = H[-1] + L[i]
                    if d <= gamma:
                        gamma = d
                    self.visualize(np.vstack((R, L)), H)
                    print(errs)
            if errs > 1000:
                error_exists = False
                print("1000 errors: no solution")
        return H, gamma, errs


# -------------------------- PERFORMANCE EVALUATION --------------------------

# --------------------------------- MAIN -------------------------------------

    def main(self, D):
        if D is None:
            D = self.gen_data()
        H = self.train(D)
        return None


N = 100
dim = 3
D = np.array([[0.25, 0.25, -1],
              [.25, .75, 1],
              [.75, .75, 1],
              [.75, .25, -1]]).T
MarginMax(N, dim)
