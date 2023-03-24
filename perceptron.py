import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)


class Perceptron(object):

    def __init__(self):

        self.name = "Perceptron Alg"
        print("Perceptron is running!\n")

    def generateData(d, n):
        pass

    def train(data):
        pass

    def evaluate(data, th):
        pass


def y_filt(y):
    if (0 <= y < 1):
        return True
    else:
        return False


def mkunitary(a, n):
    unitary = np.array([[0 for i in range(n)] for i in range(n)])
    for i in range(n):
        for j in range(n):
            if i == j:
                unitary[i][j] = a

    return unitary


# returns a populated array defining a line
def mkline(th, th0, x_min, x_max):

    if th[0][0] == 0:
        line = np.array([list(range(-5, 6, 2)), [th[1][0]] * 6])
        print(line)
    elif th[1][0] == 0:
        line = np.array([[th[0][0]] * 6, list(range(-5, 6, 2))])
    else:
        m = - th[0][0] / th[1][0]
        b = - th0 / th[1][0]

        line = np.array([[x_min, m * x_min + b],
                         [0, b], [1, b+m], [x_max, m * x_max + b]]).transpose()

    return line

def mklineslope(m, b, x_min, x_max):
    
    dx = x_max - x_min
    
    line = np.array([[x_min, m * x_min + b], [0, b], [1, b+m],
                     [x_max, m * x_max + b]]).transpose()
    
    return line


# returns an array of the signs (polarity) of the distance
# to the hyperplane theta in each dimension
def isPositive(x, th):

    # check input data for compatibility
    if (len(x[0]) != (len(th))):
        print("input matrices must be of compatible dimensions: (a x n)*(n x b)\n")
        
    dist = (th.T@x)/(th.T@th)**0.5
    return np.sign(dist)

# this function produces n randomly sampled data points and
# assigns a label to each as an added dimension, can isolate
# data using D[0:-1, :]
def generator(d, n):
    data = np.random.rand(d, n)
    labels = assignlabels(data)
    D = np.concatenate((data,labels))
    return D



# this function defines a correlation in the dataset
def assignlabels(data):
    
    n = len(data[0])
    dx = max(data[0,:]) - min(data[0,:])
    dy = max(data[1,:]) - min(data[1,:])
    labels = np.array([[0 for i in range(n)]])
    #print(labels)
    
    for i in range(n):
        
        #if data[0][i] >= dx*0.3 and data[1][i] >= dy*0.6:
        if data[1][i] >= (0.90*data[0][i] + 0.1):
            labels[0][i] = 1
        else:
            labels[0][i] = -1
            
    return labels

# create an array of colour values to visualize the labels
def colour(labels):
    n = len(labels)
    colours = np.array([0.1 for i in range(n)])
    for i in range(len(labels)):
        if labels[i] <= 0:
            colours[i] = 1.2
        else:
            colours[i] = 0.5
            
    return colours

def distance(x, th):
    return (th.T @ x) / (th.T @ th)**0.5

# this function should produce the error E(theta, theta0) for a given 
# training dataset
def evaluate(data, labels, th):
    #pos = np.array([np.sign(x) for x in labels.T*(data.T@th/(th.T@th)**0.5)])
    score = num_mistakes = 0
    for x in labels.T*(data.T@th/(th.T@th)**0.5):
        score += np.sign(x[0])
    return score

itr_num = 0


def perceptron2(x, labels, th = np.array([[0], [0]]),
               th0 = np.array([[0]]), visible = False):

    # mistakes = np.array([])

    for i in range(len(data[0, :])):
        adj_dist = labels[i] * np.sign(th.T @ x[:, i] + th0[0])
        if (adj_dist) <= 0:
            # mistakes = np.append(mistakes, i)
            th = np.array(th.T + labels[i] * x[:, i]).T
            th0 = th0 + labels[i]

            if visible:
                line = mkline(th, th0, min(x[0, :]), max(x[1, :]))
                ax.plot(line[0, :], line[1, :])
            #itr_num += 1
            th, th0 = perceptron2(x[:, i:], labels[i:], th, th0, visible)
     
       
    print("BEST theta: ", th, "\nBEST th0:", th0, "mistakes: ")         
    return th, th0


# This is a training function for dataset x (d+1, n), extra dimension contains
# the labels for dataset x (not implemented yet)
def perceptron(x, labels, th = np.array([[0], [0]]),
               th0 = np.array([[0]]), visible = False):

    exists_conflict = True
    mistakes = np.array([])

    while (exists_conflict):
        exists_conflict = False

        for i in range(len(data[0, :])):

            if (labels[i] * np.sign(th.T @ x[:, i] + th0[0])) <= 0:
                exists_conflict = True
                mistakes = np.append(mistakes, i)
                th = np.array(th.T + mkunitary(labels[i],
                                               len(x[:, 0])) @ x[:, i]).T
                th0 = th0 + labels[i]
                
                if visible:
                    line = mkline(th, th0, min(x[0, :]), max(x[1, :]))
                    ax.plot(line[0,:], line[1,:])
                    
                # th, th0 = perceptron(x[:, i:], labels, th, th0)

    print("BEST theta: ", th, "\nBEST th0:", th0, "mistakes: ", len(mistakes))         
    return th, th0

def train(data, labels):
    itrs = 10
    err = 0
    for i in len(itrs):
        # generate data
        # run perceptron on data
        # evaluate output theta classifier
        # increment error according to score
        pass

# -----------------------------------------------------------------------------
   
"""            
data = np.array([[ 1,  2], [ 1,  3], [ 2,  1], [ 1, -1], [ 2, -1]]).transpose()
labels = np.array([-1, -1, 1, 1, 1])

data = np.array([[1, -1], [0, 1], [-1.5, -1]]).transpose()
labels = np.array([1, -1, 1])

data = np.array([[-3, 2], [-1, 1], [-1, -1], [2, 2],[1, -1]]).transpose()
labels = np.array([1, -1, -1, -1, -1])

D = generator(2,50)
data = D[0:-1, :]
labels = D[-1, :]

"""
data = np.array([[-3, 2], [-1, 1], [-1, -1], [2, 2],[1, -1]]).transpose()
labels = np.array([1, -1, -1, -1, -1])

th = np.array([[0], [0]])
th0 = np.array([[0]])

clr_flags = colour(labels)

ax.set_ybound(0,1)
ax.set_xbound(0,1)

ax.scatter(data[0,:], data[1,:], c = clr_flags)

[ths, th0s] = perceptron2(data, labels, th, th0, False)
[th, th0] = perceptron(data, labels, th, th0, False)

#score = evaluate(data, labels, th)
#print("Score: ", score)

line = mkline(th, th0, min(data[0, :]), max(data[1, :]))
line_s = mkline(ths, th0s, min(data[0, :]), max(data[1, :]))
ax.plot(line[0,:], line[1,:], c = 'b')
ax.plot(line_s[0,:], line_s[1,:], c = 'b')

#line2 = mkline(3/4, 2/4, 0, 2, 100)
#ax.plot(line2[0,:], line2[1,:], c = 'b')

#line = mkline(2, -3, 0, 3, 50)
#ax.plot(line[0,:], line[1,:], c = "b")




