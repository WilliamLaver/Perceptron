#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 12:01:47 2023

                    **PLAYING WITH FEATURES**
@author: william
"""
import numpy as np
import cv2
import os



# returns array of specified depth and breadth
def gen2dArray(h, w):
    return np.array([np.random.randint(255, size=w) for i in range(h)])


# this returns a 'mirrored' padding around an image file
def genPadding(a):
    new = np.array([np.concatenate(([a[i][0]], a[i], [a[i][-1]]))
                       for i in range(len(a))])
    new = np.concatenate(([new[0]], new, [new[-1]]))
    return new


# a must be an array with 1 layer of padding and b must be a 3x3 filter
def convolution(a, b):
    conv = np.array([[0 for j in range(len(a[0])-2)] for i in range(len(a)-2)])
    for i in range(len(a) - 2):
        for j in range(len(a[0]) - 2):
            p = a[j : j + 3, i : i + 3]
            val = 0
            for k in range(len(p)):
                for l in range(len(p[0])):
                    val += (p[l][k] * b[l][k])
            conv[j][i] = val
    return conv

def scanArray(A, b):
    q = len(A) - len(b) + 1
    q0 = len(A[0]) - len(b[0]) + 1
    output = np.array([[0 for i in range(q0)] for i in range(q)])
    for i in range(q):
        for j in range(q0):
            patch = A[i:i+len(b), j:j+len(b[0])]
            out = patch @ b.T
            print(out)

def makeFilePath(fp, suffix):
    split = fp.split('/')
    np = ''
    for i in range(len(split) - 1):
        np += (split[i] + '/')
    end, ft = split[-1].split('.')
    return np + end + '/' + end + '_' + suffix + '.' + ft


def makeNewDir(fp):
    split = fp.split('/')
    np = ''
    for i in range(len(split) - 1):
        np += (split[i] + '/')
    end, ft = split[-1].split('.')
    os.mkdir(np + end + '/')
    return True


def processImage(picpath, filters):
    for filt in range(len(filters)):
        res = np.array([[[0 for j in range(len(pic))] for i in range(len(pic))]
                        for i in range(3)]).T
        for i in range(3):
            padded = genPadding(pic[:, :, i])
            res[:, :, i] = convolution(padded, filters[filt])
        newpath = makeFilePath(picpath, 'filter' + str(filt))
        print(newpath)
        cv2.imwrite(newpath, res)


# ----------------------------- EXECUTABLE --------------------------------
picpath = '/home/william/Pictures/frog.jpg'
pic = cv2.imread(picpath, 1)
makeNewDir(picpath)

# FILTERS:
b = np.array([[1, 0, -1] for i in range(3)])
c = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])             # sharpen
d = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])         # edge detection
e = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])              # emboss
f = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])              # bottom sobel
g = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])              # left sobel
h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])              # right sobel
i = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])              # top sobel
j = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125],
              [0.0625, 0.125, 0.0625]])                         # blur
filters = [b, c, d, e, f, g, i, j]

processImage(picpath, filters)
