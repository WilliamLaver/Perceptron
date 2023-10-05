#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:33:30 2023

@author: william

This program is to refresh my memory on the fundamentals of the perceptron
algorithm

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


class Perceptron(object):

    def __init__(self):
        self.purpose = "To correctly classify data points"
        self.fig = plt.figure(figsize=plt.figaspect(1))
        self.ax = self.fig.add_subplot(111, projection='3d')
        """self.ax.axes.set_xlim3d(left=0, right=1)
        self.ax.axes.set_ylim3d(bottom=0, top=1)
        self.ax.axes.set_zlim3d(bottom=0, top=1)"""

    def statePurp(self):
        print("My purpose is:\n" + self.purpose)

# ----------------------------- UTILITY FUNCTIONS --------------------------
    # calculate and return the signed distance from a point x to hyperplane th
    def distance(self, x, th, th0):
        if not th.any():
            return 0
        else:
            return (th.T @ x + th0) / (th.T @ th)**.5

    # Return the margins of error between data points and a plane separator
    def collect_margins(self, th, th0, data, labels):
        n = len(data[0, :])
        print(n)
        margins = np.array([labels[0][0] * self.distance(data[:-1, 0],
                                                         th, th0)])
        for i in range(1, n):
            m = np.array([labels[0][i] * self.distance(data[:-1, i], th, th0)])
            margins = np.concatenate((margins, m))
        return margins

    # create an array of colour values to visualize the labels
    def colourLabels(self, labels, palette=1):
        if palette == 1:
            c1 = '#d94a02'
            c2 = '#01912a'
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

    def z_func(self, x, y, a, b, c, d):
        return -1 * (a * x + b * y + d) / c
# ----------------------------- PLOTTING FUNCTIONS -------------------------

    def visualize(self, data, labels, th, th0):
        self.plotPlane(th, th0)
        colours = self.colourLabels(labels, 1)
        self.plotData(data, colours)
        return True

    def plotPlane(self, th, th0):
        N = 250
        xvals = np.linspace(0, 1, N)
        yvals = np.linspace(0, 1, N)
        X, Y = np.meshgrid(xvals, yvals)
        Z = self.z_func(X, Y, th[0][0], th[1][0], th[2][0], th0)
        self.ax.plot_wireframe(X, Y, Z)
        return True

    def plotData(self, data, colours):
        self.ax.scatter(data[0, :], data[1, :], data[2, :], c=colours)
        return

# ----------------------------- DATA GENERATION -----------------------

# function returns data points linearly separated by a random line
# dim is the number of dimensions of the data and n the number of points.
# Function returns a separable dataset with a vector of labels

    def genLinSep(self, dim, n):
        data = np.random.rand(dim, n)
        labels = np.array(np.array([[0 for i in range(n)]]))
        th = np.array([[np.random.randint(-100, 100)] for i in range(3)])
        th0 = np.array([[np.random.rand()]])

        for i in range(n):
            if self.distance(data[:, i], th, th0) <= 0:
                labels[0][i] = -1
            else:
                labels[0][i] = 1

        return data, labels

    def importImage(self):
        # produces 600x600 matrix 3 times, for red, green, blue
        pic = cv2.imread('/home/william/Pictures/circle1.png')
        pic = pic.reshape((len(pic[0])**2, 3))
        labels = np.array([[0 for i in range(len(pic))]])
        for i in range(len(pic)):
            if not (pic[i] == np.array(255)).all():
                labels[0][i] = 1
            else:
                labels[0][i] = -1
        return pic.T, labels

    # Splits the data at the specified breakpoint and shuffles the entries
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

        for i in range(n):
            if i < brk:
                d1[:, i] = data[:, args1[i]]
                l1[0][i] = labels[0][args1[i]]
            else:
                d2[:, i - brk] = data[:, args2[i - brk]]
                l2[:, i - brk] = labels[0][args2[i - brk]]

        return d1, l1, d2, l2

# -------------------------- CLASSIFICATION ---------------------------------

    def trainSeparator(self, data, labels):
        dim = 3
        cieling = 1000
        th = np.array([[0] for i in range(dim)])
        th0 = np.array([[0]])
        exists_conflict = True
        errs = 0

        while (exists_conflict):
            exists_conflict = False
            for i in range(len(data[0])):
                dist = self.distance(data[:, i], th, th0)
                if labels[0][i] * dist <= 0:
                    errs += 1
                    th = np.add(th.T, labels[0][i] * data[:, i]).T
                    th0 = th0 + labels[0][i]
                    exists_conflict = True
            if errs >= cieling:
                print("ERROR: Ceiling hit at ", cieling, " mistakes.")
                exists_conflict = False

        return th, th0


# -------------------- EXECUTABLE ---------------------------------------
percy = Perceptron()
data, labels = percy.genLinSep(3, 100)
# data, labels = percy.importImage()
th, th0 = percy.trainSeparator(data, labels)
percy.visualize(data, labels, th, th0)
