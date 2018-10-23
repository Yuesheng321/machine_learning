import scipy.io
import numpy as np
import scipy.io
from PIL import Image
datafile = "ex3data1.mat"
mat = scipy.io.loadmat(datafile)
X = mat['X']
y = np.array(mat['y']).reshape(X.shape[0],)
X = np.insert(X, 0, 1, axis=1)
#print(y[10:30])



def getDataImg(row):
    width, height = 20, 20
    square = row[1:].reshape(width, height)
    return square.T


datafile = "ex3weights.mat"
mat = scipy.io.loadmat(datafile)
Theta1, Theta2 = mat["Theta1"], mat["Theta2"]
print(Theta1.shape, Theta2.shape)


#forward propagagtion
def sigmoid(z):
    return 1./(1+np.exp(-z))

def forwardPropagation(row, Thetas):#row:1*401,Thetas:1:(25*401),2:(10,26)
    features = row
    for i in range(len(Thetas)):
        Theta = Thetas[i] #25*401
        z = np.dot(Theta, features) #1*25
        a = sigmoid(z)
        if i == len(Thetas)-1:
            return a
        a = np.insert(a, 0, 1) #1*26
        features = a


def predictNN(row, Thetas):
    classes = range(1, 11)
    output = forwardPropagation(row, Thetas)
    return classes[np.argmax(np.array(output))]


Thetas = [Theta1, Theta2]
n_correct, n_total = 0., 0.
incorrect_indices = []
for irow in range(X.shape[0]):
    n_total += 1
    if predictNN(X[irow], Thetas) == int(y[irow]):
        n_correct += 1
    else:
        incorrect_indices.append(irow)
print("%0.1f%%" % (100*(n_correct/n_total)))

for x in range(1):
    i = np.random.choice(incorrect_indices)
    img = Image.fromarray(getDataImg(X[i]) * 255)
    img.show()
    predicted_val = predictNN(X[i], Thetas)
    predicted_val = 0 if predicted_val == 10 else predicted_val
    print(predicted_val)