import numpy as np
import matplotlib.pylab as plt
fname = "data/ex2data1.txt"
data = np.loadtxt(fname, delimiter=',', usecols=(0, 1, 2), unpack=True)
X = np.transpose(np.array(data[:-1, :]))
y = np.transpose(np.array(data[-1:, :]))
m = len(y)
X = np.insert(X, 0, 1, axis=1)
print("X.shape:")
print(X.shape)
print("y.shape:")
print(y.shape)

#Visualizing the data
pos = np.where(y.reshape(m,) == 1)
neg = np.where(y.reshape(m,) == 0)
plt.figure()
plt.plot(X[pos, 1], X[pos, 2], "k+", markersize=5, label="Admitted")
plt.plot(X[neg, 1], X[neg, 2], "ro", markersize=5, label="Not admitted")
plt.xlim(30, 100)
plt.ylim(30, 100)
plt.grid(True)
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")


#implementtataion
def g(z):
    return 1./(1+np.exp(-z))


def h(theta, X):    # theat(n*1), X(m*n)
    return g(np.dot(X, theta))

#show g(z)
z = np.arange(-10, 10, 0.5)
plt.figure()
plt.plot(z, g(z))
#plt.show()

#Cost function and gradient
def costFunction(theta, X, y):
    return np.mean((-y)*np.log(h(theta, X))-(1-y)*np.log(1-h(theta, X)))

def gradient(theta, X, y):
    return (np.dot(X.T, h(theta, X) - y)/len(y))
# print("-------------")
theta = np.zeros((X.shape[1], 1))
print("theta.shape:")
print(theta.shape)
cost = costFunction(theta, X, y)
grad = gradient(theta, X, y)
print("cost:")
print(cost)
print("gradient:")
print(grad)

# test_theta = np.array((-24, 0.2, 0.2)).reshape(3, 1)
# cost = costFunction(test_theta, X, y)
# grad = gradient(test_theta, X, y)
# print("cost:")
# print(cost)
# print("gradient:")
# print(grad)


from scipy import optimize
def optimizeThete(theta, X, y):
    result = optimize.fmin(costFunction, x0=theta, args=(X, y), maxiter=400, full_output=True)
    return result[0], result[1]