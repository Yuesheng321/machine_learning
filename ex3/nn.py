import numpy as np
import scipy.io
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

datafile = "ex3data1.mat"
mat = scipy.io.loadmat(datafile)
#print(mat['X'])
X = mat['X']
y = np.array(mat['y']).reshape(X.shape[0],)
X = np.insert(X, 0, 1, axis=1)
# np.set_printoptions(threshold=np.inf)
print(X.shape, y.shape)
print(y[0:10])

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
    img.show()

displayData()

def sigmoid(z):
    return 1./(1+np.exp(-z))
def h(theta, X):
    return sigmoid(np.dot(X, theta))
def cost(theta, X, y):
    m = X.shape[0]
    hx = h(theta, X)
    term1 = -y*np.log(hx)
    term2 = (1-y)*np.log(1-hx)