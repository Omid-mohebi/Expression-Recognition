import numpy as np
import pandas as pd

def get_data():
    raw_data = pd.read_csv('../fer2013.csv')

    # Normalizing
    X = np.array([np.fromstring(x, sep=' ') for x in raw_data['pixels']]) / 255.0
    Y = raw_data['emotion'].astype(int).to_numpy()

    # Balancing
    X1, Xelse = X[Y == 1], X[Y != 1]
    Yelse = Y[Y != 1]
    X1 = np.repeat(X1, 9, axis=0)
    X = np.vstack([Xelse, X1])
    Y = np.concatenate([Yelse, np.ones(len(X1))])

    # One_hard
    classes = set(Y)
    temp_y = np.zeros((len(Y), classes))
    for i in len(Y):
        temp_y[i, Y[i]] = 1
    Y = temp_y

    # Shuffling
    indices = np.random.permutation(len(Y))
    X, Y = X[indices], Y[indices]
    
    # Splitting the data
    split_border = int(len(Y) * 0.7)
    Xtrain, Ytrain = X[:split_border], Y[:split_border]
    Xtest, Ytest = X[split_border:], Y[split_border:]

    return Xtrain, Ytrain, Xtest, Ytest


# Model and helper functions

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def softmax(X):
    exps = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exps / exps.sum(axis=1, keepdims=True)

def relu(X):
    return np.maximum(0, X)

def forward(X, w1, b1, w2, b2):
    z = relu(X.dot(w1) + b1)
    return z, softmax(z.dot(w2) + b2)

# Hyperparameters
learning_rate = 0.001
epochs = 1000
reg = 0.01

# Load data
Xtrain, Ytrain, Xtest, Ytest = get_data()
N, D = Xtrain.shape
M = 10
K = len(set(Ytrain))

# Initialize weights and biases
w1 = np.random.randn(D, M) / np.sqrt(D + M)
b1 = np.zeros(M)
w2 = np.random.randn(M, K) / np.sqrt(M + K)
b2 = np.zeros(K)

# Training loop
for i in range(epochs):
    z, py = forward(Xtrain, w1, b1, w2, b2)
    py_y = py - np.eye(K)[Ytrain]

    # Update weights and biases for w2 and b2
    w2 -= learning_rate * (z.T.dot(py_y) + reg * w2)
    b2 -= learning_rate * (py_y.sum(axis=0) + reg * b2)

    # Backpropagation for hidden layer
    dz = py_y.dot(w2.T) * (z * (1 - z))
    w1 -= learning_rate * (Xtrain.T.dot(dz) + reg * w1)
    b1 -= learning_rate * dz.sum(axis=0)