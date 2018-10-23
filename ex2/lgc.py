import numpy as np
import matplotlib.pylab as plt
fname = "data/ex2data1.txt"
data = np.loadtxt(fname, delimiter=',', usecols=(0, 1, 2), unpack=True)
X = np.transpose(np.array(data[:-1, :]))
y = np.array(data[-1:, :]).reshape(X.shape[0],)
m = len(y)
X = np.insert(X, 0, 1, axis=1)
#Visualizing the data
def plotData():
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.figure()
    plt.plot(X[pos, 1], X[pos, 2], "k+", markersize=5, label="Admitted")
    plt.plot(X[neg, 1], X[neg, 2], "ro", markersize=5, label="Not admitted")
    plt.xlim(30, 100)
    plt.ylim(30, 100)
    plt.grid(True)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")


plotData()
plt.show()
#implementataion
def g(z):
    return 1./(1+np.exp(-z))


def h(theta, X):    # theat(1*n), X(m*n)
    return g(np.dot(X, theta))

#show g(z)
z = np.arange(-10, 10, 0.5)
plt.figure()
plt.plot(z, g(z))
plt.show()

#Cost function and gradient
def costFunction(theta, X, y):
    return np.mean((-y)*np.log(h(theta, X))-(1-y)*np.log(1-h(theta, X)))

def gradient(theta, X, y):
    return np.dot(X.T, h(theta, X) - y)/len(y)

theta = np.zeros((X.shape[1],))
#print(np.dot(X, theta))
print("theta.shape:")
print(theta.shape)
cost = costFunction(theta, X, y)
grad = gradient(theta, X, y)
print("cost:")
print(cost)
print("gradient:")
print(grad)
from scipy import optimize
def optimizeThete(theta, X, y):
    return optimize.fmin_tnc(costFunction, x0=theta, fprime=gradient, args=(X, y))


result = optimizeThete(theta, X, y)
theta = result[0]
print(costFunction(theta, X, y))

boundary_x = np.array((np.min(X[:, 1]), np.max(X[:, 1])))
boundary_y = (-1./theta[2])*(theta[0] + theta[1]*boundary_x)
plotData()
plt.plot(boundary_x, boundary_y, "b-", label="Decision Boundary")
plt.show()


def predict(theta, X):
    probability = h(theta, X)
    return [1 if x >= 0.5 else 0 for x in probability]


prediction = predict(theta, X)
correct = [1 if a == b else 0 for (a, b) in zip(prediction, y)]
accuracy = sum(correct)/len(y)
print("accuracy:%f" % accuracy)