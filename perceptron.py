#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 20:30:52 2023

@author: william
"""
import numpy as np
import matplotlib.pyplot as plt


class Perceptron(object):

    def __init__(self):
        self.purpose = "To correctly classify data!"
        self.purpose_sub = "To serve my master."
        self.Fig = plt.figure(figsize=plt.figaspect(1))
        self.ax = self.Fig.add_subplot(111)

        print("I AM PERCEPTRON.\nREADY TO CLASSIFY DATA!\n")

#                           PLOTTING FUNCTIONS
# ----------------------------------------------------------------------------

    def plotline(self, th, th0, x_min, x_max):
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
        self.ax.plot(line[0, :], line[1, :])
        return True


#                            UTILITY FUNCTIONS:
# -----------------------------------------------------------------------------

    # returns column vector of zeroes, length k, xth row value is 1
    def one_hot(self, x, k):
        return np.array([[0] if (x - 1) != i else [1] for i in range(k)])

    def identity(self, a, n):
        identity = np.array([[0 for i in range(n)] for i in range(n)])
        for i in range(n):
            for j in range(n):
                if i == j:
                    identity[i][j] = a
        return identity

    # Normalized, signed distance to the hyperplane.
    def distance(self, th, th0, x):
        if not th.any():
            return 0
        return (th.T @ x + th0) / (th.T @ th)**.5

    # create an array of colour values to visualize the labels
    def colour(self, labels, palette=1):
        if palette == 1:
            c1 = '#1821b5'
            c2 = '#d05703'
        else:
            c1 = '#0db902'
            c2 = '#990170'
        n = len(labels[0, :])
        colours = np.array([c1 for i in range(n)])
        for i in range(n):
            if labels[0][i] <= 0:
                colours[i] = c1
            else:
                colours[i] = c2
        return colours

    def data_split(self, data, labels, q=0.8):
        n = len(data[0, :])
        brk = int(q * n)

        d1 = np.array([[0.for i in range(brk)] for i in range(len(data))])
        d2 = np.array([[0. for i in range(n - brk)] for i in range(len(data))])

        l1 = np.array([[0. for i in range(brk)]])
        l2 = np.array([[0. for i in range(n - brk)]])

        args1 = np.array([range(brk)])[0]
        np.random.shuffle(args1.T)
        args2 = np.array([range(brk, n)])[0]
        np.random.shuffle(args2.T)
        print(len(args1), "\n", args1)
        print(len(args2), "\n", args2)
        for i in range(n):
            print(brk, ", ", i)
            if i < brk:
                d1[:, i] = data[:, args1[i]]
                l1[0][i] = labels[0][args1[i]]
            else:
                d2[:, i - brk] = data[:, args2[i - brk]]
                l2[:, i - brk] = labels[0][args2[i - brk]]
            print(d1)
        return d1, l1, d2, l2
#                           DATA GENERATION
# -----------------------------------------------------------------------------

    def generate_linsep(self, dim, n):
        data = np.random.rand(dim, n)
        labels = self.assignlinear(data)
        D = np.concatenate((data, labels))
        return data, labels

    # this function defines a linear correlation in the dataset
    def assignlinear(self, data):
        n = len(data[0])
        dx = max(data[0, :]) - min(data[0, :])
        dy = max(data[1, :]) - min(data[1, :])
        m = np.random.rand()
        b = np.random.rand()
        labels = np.array([[0 for i in range(n)]])

        for i in range(n):
            # if data[0][i] >= dx*0.3 and data[1][i] >= dy*0.6:
            # if (data[0][i]**2 + data[1][i]**2)**0.5 <= 7:
            if data[1][i] >= (m * data[0][i] + b):
                labels[0][i] = 1
            else:
                labels[0][i] = -1

        return labels

#                           CLASSIFICATION
#-----------------------------------------------------------------------------


    def train(self, data, labels):
        gamma = None
        ceiling = 100000
        errors = 0
        th = np.array([[0], [0]])
        th0 = 0

        exists_conflict = True

        while (exists_conflict):
            exists_conflict = False

            for i in range(len(data[0, :])):
                dist = self.distance(th, th0, data[:, i])
                if gamma is None or dist < gamma:
                    gamma = labels[0][i] * dist
                if dist * labels[0][i] <= 0:
                    exists_conflict = True
                    errors += 1
                    th = np.add(th.T, labels[0][i] * data[:, i]).T
                    th0 = th0 + labels[0][i]
            if errors > ceiling:
                exists_conflict = False

        print("n_errs: ", errors, "gamma: ", gamma, "\ntheta: ",
              th, "\ntheta_0: ", th0)

        return th, th0

    def classify(self, data, th, th0):
        n = len(data[0, :])
        labels = np.array([[0 for i in range(n)]])
        for i in range(n):
            labels[0][i] = np.sign(self.distance(th, th0, data[:, i]))

        return labels

#                               EXECUTABLE
# ----------------------------------------------------------------------------


percy = Perceptron()
data, labels = percy.generate_linsep(2, 50)
d_train, l_train, d_test, l_test = percy.data_split(data, labels, q=0.8)
colours1 = percy.colour(l_train, palette=1)
colours2 = percy.colour(l_test, 2)
percy.ax.scatter(d_train[0, :], d_train[1, :], c=colours1)
percy.ax.scatter(d_test[0, :], d_test[1, :], c=colours2)
th, th0 = percy.train(d_train, l_train)
percy.plotline(th, th0, 0, 1)
percy.Fig
