#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:36:36 2024

@author: william
"""

import numpy as np
import matplotlib.pyplot as plt
import time


class Classifier(object):

    def __init__(self, N, dim, q=0.3):
        self.N = N
        self.dim = dim
        self.q = q

        return None

# ----------------------------- UTILITIES ------------------------------------

    # returns a signed distance to the point x from plane th
    def dist(self, x, H):
        if not H.any():
            return 0
        return np.dot(H[:-1].T, x) + H[-1]

    def sig(self, z):
        return (1 + np.e**(-z))**(-1)

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
        self.ax.plot(x, y, c='r')
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
        D = np.array([np.random.rand(self.N) for i in range(self.dim)])
        D = self.separate(D)
        return D

    def splitData(self, D, q):
        D = D.T
        A = np.array([[0.0 for j in range(self.dim + 1)]
                      for i in range(int(q * len(D)))])
        B = np.array([[0.0 for j in range(self.dim + 1)]
                      for i in range(int((1-q) * len(D)))])
        for i in range(len(A)):
            k = np.random.randint(0, len(D))
            A[i] = D[k]
            if k == 0:
                D = D[1:]
            else:
                D = np.vstack((D[:k], D[k+1:]))
        for i in range(len(B)):
            if len(D) == 1:
                k = 0
            else:
                k = np.random.randint(0, len(D))
            B[i] = D[k]
            if k == 0:
                D = D[1:]
            else:
                D = np.vstack((D[:k], D[k+1:]))
        return A.T, B.T

# -----------------------LEARNING ALGORITHMS----------------------------------

    def perceptron(self, D, T=1000, visual=False):
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
                    if visual:
                        self.visualize(np.vstack((R, L)), H)
            if errs > T:
                error_exists = False
                print(str(T) + " errors => no solution found")
        return H

    def averagedPerceptron(self, D, T=1000, visual=False):
        H = np.array([[0.0] for i in range(self.dim + 1)])
        R = D[:-1]
        L = D[-1]
        gamma = 1
        errs = 0
        Hs = H.copy()
        itrs = 0
        error_exists = True
        while (error_exists):
            error_exists = False
            for i in range(len(R[0])):
                itrs += 1
                d = self.dist(R[:, i], H)
                if d * L[i] <= 0:
                    error_exists = True
                    errs += 1
                    H[:-1] = np.add(H[:-1].T, L[i] * R[:, i]).T
                    H[-1] = H[-1] + L[i]
                    if d <= gamma:
                        gamma = d
                    if visual:
                        self.visualize(np.vstack((R, L)), H)
                Hs = np.add(Hs, H)
            if errs > T:
                error_exists = False
                print(str(T) + " errors => no solution found")
        return Hs / itrs

    def gradientDescent(self, D, T=1000, visual=False):
        def obj_func(Data, H, lam):
            def hinge(v):
                return np.where(v >= 1, 0, 1 - v)
            D, L = Data[:-1], np.array([Data[-1]])
            return np.mean(hinge(L * self.dist(D, H))) + lam * (H[:-1].T @ H[:-1])

        def grad_obj_func(Data, H, lam):
            def d_hinge(v):
                return np.where(v >= 1, 0, -1)
            D, L = Data[:-1], np.array([Data[-1]])
            dth = np.mean(d_hinge(L * self.dist(D, H)) * L * D, axis=1, keepdims=True) + 2 * lam * H[:-1]
            dth0 = np.mean(d_hinge(L * self.dist(D, H)) * L, axis=1, keepdims=True)
            return np.vstack((dth, dth0))

        def eta(t):
            return 2 / (t + 1)**.5

        H = np.array([[0.] for i in range(self.dim + 1)])
        H_p = H.copy()
        eps = .0000001
        lam = .0001
        min_found = False
        t = 0
        while (not min_found):
            H = H - eta(t) * grad_obj_func(D, H, lam)
            if visual:
                self.visualize(D, H)
            diff = np.abs(obj_func(D, H, lam) - obj_func(D, H_p, lam))
            if diff < eps or t >= T:
                min_found = True
            H_p = H.copy()
            t += 1
        return H

    def SGD(self, D, T=1000, visual=False):
        def grad_obj_func(x, L, H, lam):
            def d_hinge(v):
                return np.where(v >= 1, 0, -1)
            dth = (d_hinge(L * self.dist(x, H)) * L * x + 2 * lam * H[:-1].T).T
            dth0 = d_hinge(L * self.dist(x, H)) * L
            return np.vstack((dth, dth0))

        def eta(t):
            return 2 / (1 + t)**.5
        
        H = np.array([[0.] for i in range(self.dim + 1)])
        Hp = H.copy()
        lam = 0.001
        t = 0
        for t in range(1, T):
            i = np.random.randint(0, len(D[0]))
            H = H - eta(t) * grad_obj_func(D[:-1][:, i], D[-1][i], H, lam)
            H_p = H.copy()
            t += 1
        return H

# -------------------------- PERFORMANCE EVALUATION --------------------------

    def evaluate_accuracy(self, D, H):
        score = 0.
        R = D[:-1]
        L = D[-1]
        for i in range(len(R[0])):
            if self.dist(R[:, i], H) * L[i] >= 0:
                score += 1
        return score / len(R[0])

    def evaluate_algorithm(self, classifier, T, q=0.3):
        # self.evaluate_robustness()
        score = 0.
        speed = 0
        for i in range(T):
            D = self.gen_data()
            train, test = self.splitData(D, q)
            st = time.time()
            H = classifier(train)
            ft = time.time()
            speed += (ft - st)
            score += self.evaluate_accuracy(test, H)
        print(str(T) + " iterations: " + str(100 * score / T) + "% Accurate")
        print("Average iteration complete in: " + str(round(speed / T, 3)) + "s")
        return score / T

    def evaluate_robustness(self):
        D = self.gen_data()
        q = 0.01
        acc = np.array([0.0 for i in range(98)])
        qs = np.arange(0, .98, 0.01)
        for i in range(len(acc)):
            train, test = self.splitData(D, q)
            H = self.train(train)
            score = self.evaluate_accuracy(test, H)
            print(str(score * 100) + "% accurate")
            acc[i] = score
            q += .01
        plt.clf()
        ax = self.fig.add_subplot(111)
        ax.plot(qs, acc)
        plt.show()
        return None

# --------------------------------- MAIN -------------------------------------

    def main(self, D=None, classifier=None, visual=False):
        if classifier is None:
            classifier = self.perceptron
        if D is None:
            D = self.gen_data()
        if self.q < 1.0:
            train, test = self.splitData(D, self.q)
        else:
            train = D
            test = train.copy()
        
        H = classifier(train, 10000, visual)
        self.visualize(test, H)
        score = self.evaluate_accuracy(test, H)
        print(str(score * 100) + "% accurate, trained on " + str(self.q*100) + " % of the data")
        print("H = " + str(H.T))
        return H

# --------------------------- EXECUTABLE ------------------------------------

N = 10000
dim = 2
D = np.array([[0.25, 0.25, -1],
              [.25, .75, 1],
              [.75, .75, 1],
              [.75, .25, -1]]).T
classifier = Classifier(N, dim, .3)
classifier.main(None, classifier.gradientDescent, False)
#classifier.evaluate_algorithm(classifier.perceptron, 10, 0.3)
#classifier.evaluate_algorithm(classifier.averagedPerceptron, 10, 0.3)
#classifier.evaluate_algorithm(classifier.gradientDescent, 10, 0.3)
#classifier.evaluate_algorithm(classifier.SGD, 10, 0.3)

