import numpy as np
import matplotlib.pyplot as plt

def computeCost(x, y, theta):
    m = len(y)
    #print(m)
    J = 0
    #print(len(np.dot(x, theta)))
    J = sum((np.dot(x, theta) - y) ** 2)/float(2 * m) #x(m, 2),theta(2, 1)
    return J

def gradientDescent(x, y, theta, alpha, iterations):
    m = len(y) #样本数
    J_history = np.zeros([iterations, 1])
    for iter in range(iterations):
        xt = x.T
        #print(sum(np.dot(xt, (np.dot(x, theta) - y))))
        theta = theta - (alpha/m)*(np.dot(xt, (np.dot(x, theta) - y)))
        #print(theta)
        J_history[iter] = computeCost(x, y, theta)
    return theta, J_history

        #置与theta可看到每250次后训练的模型
        #if iter%250 == 0:
        #    plt.plot(x[:, 1], np.dot(x, theta), "-", c="r")
######parting1: plotting
print("Plotting Data ...\n")
data = np.loadtxt("ex1data1.txt", delimiter=",")
#print(data)
x = data[:, 0]  #(m,)
y = data[:, 1]  #(m,)
m = len(y) #样本数
x = x.reshape([m, 1])   #(m,1)
y = y.reshape([m, 1])   #(m,1)
#print(x)
#print(y)

plt.figure()
plt.scatter(x, y, c="r", marker="x")
plt.xlabel("population of city in 10000s")
plt.ylabel("profit in $10000s")
#plt.show()

#####part2:Gradent desent
x = np.concatenate((np.ones([m, 1]), x), axis=1)
#print(x)
theta = np.zeros([2, 1])
#hθ(x) = θT x = θ0 + θ1x1

iterations = 1500
alpha = 0.01
print(computeCost(x, y, theta))
#test another
#theta = np.array([[-1], [2]])
#print(computeCost(x, y, theta))

print("\n Running Gradient Descent...\n")
theta, J_history = gradientDescent(x, y, theta, alpha, iterations)
print("Theta found by gradient descent:")
print(theta[0, 0], theta[1, 0])
plt.plot(x[:, 1], np.dot(x, theta), "-")
plt.legend(["Liner regression", "Training Data"])
plt.show()
# plt.figure()
# print(J_history)
# J_len = len(J_history)
# plt.plot(range(0, J_len), J_history)
# plt.show()

predict1 = np.dot(np.array([1, 3.5]), theta)
predict2 = np.dot(np.array([1, 7]), theta)
print("For population = 35000, we predict a profit of %f\n" % (predict1*10000))
print("For population = 70000, we predict a profit of %f\n" % (predict2*10000))


#####Visualizing J(theta0. theta1)
print("Visualizing J(theta0, theta1)...\n")
theta0_vals = np.linspace(-10, 10, 100)
#print(theta0_vals)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        # every values of theta0 and theta1
        t = np.array([theta0_vals[i], theta1_vals[j]]).reshape([2, 1])
        J_vals[i, j] = computeCost(x, y, t)
#J_vals = J_vals.reshape([10000, ])
#print(J_vals.shape)
#J_vals = J_vals.T
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
# print(theta0_vals)
# print(theta1_vals)
theta1_vals, theta0_vals = np.meshgrid(theta1_vals, theta0_vals)
# print(theta0_vals)
# print(theta1_vals)
# print(theta0_vals.shape, theta1_vals.shape, J_vals.shape)
#rstride和cstride表示行列隔多少个取样点建一个小面，cmap表示绘制曲面的颜色
ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1, cmap="rainbow")
plt.xlabel("theta_0")
plt.ylabel("theta_1")
plt.show()

#plt.contourf(theta0_vals, theta1_vals, J_vals, 20, alpha=0.6, cmp=plt.cm.hot)
plt.contour(theta0_vals, theta1_vals, J_vals, 20, colors="black")
plt.plot(theta[0, 0], theta[1, 0], 'r', marker='x', markerSize=10, LineWidth=2)
plt.show()