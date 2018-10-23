import numpy as np
#1
# x = np.ones((2, 3))
# print(x)
# y = np.ones((2, 1))
# print(y)
# z = np.concatenate((x, y), axis=1)
# print(z)

# #2
# x = np.arange(2*2*4).reshape([2, 2, 4])
# y = np.arange(2*2*4).reshape([2, 4, 2])
# #print(x)
# #print(y)
# #print(np.matmul(x, y)) #(2, 2, 2)
# np.matmul(a, b)[0, 1, 1]
# sum(a[0, 1, :] * b[0, :, 1])

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = [1, 1, 2, 2]
# Y = [3, 4, 4, 3]
# Z = [1, 2, 1, 1]
# ax.plot_trisurf(X, Y, Z)
# plt.show()

# import matplotlib.pyplot as plt  # 绘图用的模块
# from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数
# import numpy as np
#
#
# def fun(x, y):
#     return np.power(x, 2) + np.power(y, 2)
#
#
# fig1 = plt.figure()  # 创建一个绘图对象
# ax = Axes3D(fig1)  # 用这个绘图对象创建一个Axes对象(有3D坐标)
# X = np.arange(-2, 2, 0.1)
# Y = np.arange(-2, 2, 0.1)  # 创建了从-2到2，步长为0.1的arange对象
# # 至此X,Y分别表示了取样点的横纵坐标的可能取值
# # 用这两个arange对象中的可能取值一一映射去扩充为所有可能的取样点
# X, Y = np.meshgrid(X, Y)
# Z = fun(X, Y)  # 用取样点横纵坐标去求取样点Z坐标
# plt.title("This is main title")  # 总标题
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)  # 用取样点(x,y,z)去构建曲面
# ax.set_xlabel('x label', color='r')
# ax.set_ylabel('y label', color='g')
# ax.set_zlabel('z label', color='b')  # 给三个坐标轴注明
# plt.show()  # 显示模块中的所有绘图对象

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fun(x, y):
    return np.power(x, 2) + np.power(y, 2)

# fig1 = plt.figure()
# ax = Axes3D(fig1)
#
# fig2 = plt.figure()
# ax = Axes3D(fig2)
# plt.show()
fig1 = plt.figure()
ax = Axes3D(fig1)
X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)
# print(X)
# print(Y)
X, Y = np.meshgrid(X, Y)
# print(X)
# print(Y)
Z = fun(X, Y)
print(Z.shape)
plt.title("main title")
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)
ax.set_xlabel("x label", color='r')
ax.set_ylabel("y label", color='g')
ax.set_zlabel("z label", color='b')
#plt.savefig('fig.png', bbox_inches='tight')
plt.show()