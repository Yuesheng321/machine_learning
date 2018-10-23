import numpy as np
import matplotlib.pyplot as plt
fname = "data/ex2data2.txt"
data = np.loadtxt(fname, delimiter=',', usecols=(0, 1, 2), unpack=True)
print("data.shape:(%s,%s)"%data.shape)
X = np.transpose(np.array(data[:-1,]))
X = np.insert(X, 0, 1, axis=1)
print("X.shape:(%s,%s)"%X.shape)
y = np.array(data[-1:,]).reshape((X.shape[0],))
print("y.shape:(%s,)"%y.shape)


#Visualizing the data
def plotData():
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.figure()
    plt.plot(X[pos, 1], X[pos, 2], "b+", markersize=5, label="y=1")
    plt.plot(X[neg, 1], X[neg, 2], "ko", markersize=5, label="y=0")
    plt.xlim(-1, 1.5)
    plt.ylim(-0.8, 1.2)
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")

plotData()
#plt.show()

def mapFeature(x1, x2):
    degrees = 6
    out = np.ones((x1.shape[0], 1))
    for i in range(1, degrees+1):   #not include degrees+1
        for j in range(0, i+1):
            term1 = x1 ** (i-j)
            term2 = x2 ** j
            term = (term1*term2).reshape(x1.shape[0], 1)
            out = np.hstack((out, term))
    return out


mappedX = mapFeature(X[:,1],X[:,2])
print("mappedX.shape:(%s,%s)"%mappedX.shape)

#Cost function and gradient
def g(z):
    return 1./(1+np.exp(-z))


def h(theta, X):
    return g(np.dot(X, theta))


def costFunction(theta, X, y, mylambeda=0):
    term1 = -y*np.log(h(theta, X)) #shape:(118,)
    term2 = (1-y)*np.log(1-h(theta, X))
    regterm = (mylambeda/2.) * np.sum(theta[1:]*theta[1:])
    return np.mean(term1 - term2 + regterm)



def gradient(theta, X, y, mylambeda=0):
    reg = mylambeda * theta
    reg[0] = 0
    return (np.dot(X.T, h(theta, X) - y) + reg)/len(y)


theta = np.zeros((mappedX.shape[1],))
# cost = costFunction(theta, mappedX, y, 1)
# print(cost)
# grad = gradient(theta, mappedX, y, 1)
# print(grad)

#Learning parameters using fmin_tnc
from scipy import optimize
def optmizeRegularizedTheta(theta, X, y, mylambeda=0):
    result = optimize.fmin_tnc(costFunction, x0=theta, fprime=gradient, args=(X, y, mylambeda))
    return result


result = optmizeRegularizedTheta(theta, mappedX, y, 1)
#print(result)
#print(costFunction(result[0], mappedX, y, 1))

#Plotting the decision boundary
from mpl_toolkits.mplot3d import Axes3D
def plotBoundary(theta):
    x1 = np.linspace(-1, 1.5, 50)
    x2 = np.linspace(-0.8, 1.2, 50)
    z = np.zeros((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            myfeature = mapFeature(np.array([x1[i]]), np.array([x2[j]]))
            z[i][j] = np.dot(myfeature, theta)
    z = z.T
    #x2, x1 = np.meshgrid(x2, x1)
    plotData()
    #print(z)
    plt.contour(x1, x2, z, 0)
   # fig = plt.figure()
   # ax = Axes3D(fig)
   # ax.plot_surface(x1, x2, z, rstride=1, cstride=1, cmap="rainbow")


theta = result[0]
plotBoundary(theta)
plt.show()