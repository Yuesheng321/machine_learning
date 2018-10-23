import scipy.io
import numpy as np
import scipy.io
import random
from PIL import Image
import itertools
datafile = "ex4data1.mat"
mat = scipy.io.loadmat(datafile)
X = mat['X']
y = np.array(mat['y']).reshape(X.shape[0],)
X = np.insert(X, 0, 1, axis=1)


def getDataImg(row):
    width, height = 20, 20
    square = row[1:].reshape(width, height)
    return square.T


def displayData():
    width, height = 20, 20
    nrows, ncols = 10, 10
    indices_to_display = random.sample(range(X.shape[0]), nrows * ncols)

    big_picture = np.zeros((height * nrows, width * ncols))
    irow, icol = 0, 0
    for idx in indices_to_display:
        if icol == ncols:
            irow += 1
            icol = 0
        iimg = getDataImg(X[idx])
        big_picture[irow * height:irow * height + iimg.shape[0], icol * width:icol * width + iimg.shape[1]] = iimg
        icol += 1
    img = Image.fromarray(big_picture*255)
    #img.show()

displayData()


#model representation
datafile = "ex4weights.mat"
mat = scipy.io.loadmat(datafile)
Theta1, Theta2 = mat["Theta1"], mat["Theta2"]
print(Theta1.shape, Theta2.shape)

input_layer_size = 400
hidden_layer_size = 25
out_layer_size = 10
n_training_samples = X.shape[0]

def flattenParams(thetas_list):
    flattened_list = [mytheta.flatten() for mytheta in thetas_list]
    combined = list(itertools.chain.from_iterable(flattened_list))
    assert len(combined) == (input_layer_size + 1)*hidden_layer_size + \
           (hidden_layer_size + 1)*out_layer_size
    return np.array(combined).reshape((len(combined), 1))


def reshapeParams(flattened_array):
    theta1 = flattened_array[:(input_layer_size + 1) * hidden_layer_size] \
            .reshape((hidden_layer_size, input_layer_size + 1))
    theta2 = flattened_array[(input_layer_size + 1) * hidden_layer_size:] \
            .reshape((out_layer_size, hidden_layer_size + 1))
    return [theta1, theta2]


def flattenX(myX):
    return np.array(myX.flatten()).reshape((n_training_samples*(input_layer_size + 1), 1))


def reshapeX(flattenedX):
    return np.array(flattenedX).reshape((n_training_samples, input_layer_size + 1))

def sigmoid(z):
    return 1./(1+np.exp(-z))
def propageteForward(row, Thetas):
    features = row
    zs_as_per_layer = []
    for i in range(len(Thetas)):
        Theta = Thetas[i]
        z = np.dot(Theta, features)
        a = sigmoid(z)
        zs_as_per_layer.append((z, a))
        if i == len(Thetas)-1:
            return np.array(zs_as_per_layer)
        a = np.insert(a, 0, 1)
        features = a


def cost(mythetas_flattened, myX_flattened, myy):   #myy(5000,)
    mythetas = reshapeParams(mythetas_flattened)
    myX = reshapeX(myX_flattened)
    total_cost = 0.
    m = n_training_samples
    for irow in range(m):
        myrow = myX[irow]
        myhs = propageteForward(myrow, mythetas)[-1][1] #activation
        tmpy = np.zeros((10,))
        tmpy[myy[irow] - 1] = 1 #shape(1,) to shape(10,)
        mycost = -np.dot(np.log(myhs), tmpy) - np.dot(np.log(1-myhs), (1-tmpy))
        total_cost += mycost
    total_cost = float(total_cost) / m
    return total_cost


myThetas = [Theta1, Theta2]
print(cost(flattenParams(myThetas), flattenX(X), y))


#Backpropagation
def sigmoidGradient(z):
    return sigmoid(z) * (1-sigmoid(z))


#print(sigmoidGradient(0))
def genRandThetas():
    epsilon_init = 0.12
    theta1_shape = (hidden_layer_size, input_layer_size+1)
    theta2_shape = (out_layer_size, hidden_layer_size+1)
    rand_thetas = [np.random.rand(*theta1_shape) * 2 * epsilon_init - epsilon_init, \
                   np.random.rand(*theta2_shape) * 2 * epsilon_init - epsilon_init]
    return rand_thetas


def backPropagation(mythetas_flattened, myX_flattened, myy):
    mythetas = reshapeParams(mythetas_flattened)
    myX = reshapeX(myX_flattened)
    Delta1 = np.zeros((hidden_layer_size, input_layer_size+1))
    Delta2 = np.zeros((out_layer_size, hidden_layer_size+1))
    m = n_training_samples
    for irow in range(m):
        myrow = myX[irow]
        a1 = myrow #  .reshape((input_layer_size+1,))   #shape(401,)
        temp = propageteForward(a1, mythetas)
        z2 = temp[0][0]
        a2 = temp[0][1]  #shape(25,)
        z3 = temp[1][0]
        a3 = temp[1][1]
        tmpy = np.zeros((10,))
        tmpy[myy[irow] - 1] = 1
        delta3 = a3 - tmpy  #(10,)
        delta2 = np.dot(mythetas[1].T[1:, :],delta3) * sigmoidGradient(z2)       #(25,)
        a2 = np.insert(a2, 0, 1)    #shape(26,)
        Delta1 += np.dot(delta2.reshape(len(delta2), 1), a1.reshape(1, len(a1)))
        Delta2 += np.dot(delta3.reshape(len(delta3), 1), a2.reshape(1, len(a2)))

    D1 = Delta1/float(m)
    D2 = Delta2/float(m)
    return flattenParams([D1, D2]).flatten()


flattenedD1D2 = backPropagation(flattenParams(myThetas), flattenX(X), y)
D1, D2 = reshapeParams(flattenedD1D2)
print(D1.shape)
print(D2.shape)
