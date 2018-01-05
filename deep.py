import pickle
import numpy as np
import matplotlib as plt
from load import load
import time

def one_of_k(data):
    labels = np.zeros((data.size, 10))
    for arg, label in enumerate(labels):
        label[data[arg]] = 1
    return labels

def softmax(x):
    dev = np.sum(np.exp(x), axis=0)
    #print("softmax shape", dev.shape)
    dev.shape = [1, dev.size]
    return np.exp(x) / dev
    
def ReLu(x):
    mask = x < 0
    dx = x[mask] = 0
    return dx

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_dif(z):
    return z * (1 - z)

def forward(X, Node, W, B, h):
    N = len(X)
    print("N",N)
    L = len(Node)
    Z = [np.zeros([Node[l], N]) for l in range(L)] # value after affine
    U = [np.zeros([Node[l], N]) for l in range(L)] # value after activation
    Z[0] = X.reshape([Node[0], N])
    for l in range(1, L):
        U[l] = np.dot(W[l].T, Z[l-1]).reshape([Node[l],N]) + B[l]
        Z[l] = h[l](U[l])
    #print("forward U[L-1].shape", U[L-1].shape)
    #print("forward Z", Z[L-1][:,0])
    #print("softmax", softmax(U[L-1])[:,0])
    return U, Z

def forward_test():
    X = np.array(
        [[0, 1, 2],
         [3, 4, 5]]).T
    Node = [2, 1, 3]
    W = {
        1: np.array([[1],
                     [2]]),
        2: np.array([[0, 1, 2]])
    }
    B = {
        1: np.array([[1]]),
        2: np.array([[0],
                     [1],
                     [2]])
    }
    h = {
        1: lambda x: x,
        2: lambda x: x
    }
    U, Z = forward(X, Node, W, B, h)
    print("forward test {}".format(Z))
forward_test()

def accuracy(X, t, Node, W, B, h):
    _, Z = forward(X, Node, W, B, h)
    L = len(Node)
    #print(Z[L-1].shape)
    arg = np.argmax(Z[L-1], axis=0)
    N = len(t)
    return np.sum(arg == t) / N


if __name__ == "__main__":
    dataset = load.load()
    t_train = one_of_k(dataset["train_labels"])
    t_test = dataset["test_labels"]
    X_train = dataset["train_images"]/255
    X_test = dataset["test_images"]/255

    print(t_train.shape)
    print(X_train.shape)
    N = 60000 # Number of data
    print(N)


    #Hyper Param
    M = 3 # maximum iterations (epoch)
    e = 0.0001 # learning rate

    Node = [784, 10]
    L = len(Node)

    # initialize
    W = {}
    B = {}
    h = {}
    h_dif = sigmoid_dif
    for l in range(1, L):
        W[l] = np.random.normal(0, 0.1, (Node[l-1], Node[l])) #weight
        B[l] = np.zeros([Node[l], 1]) # bias for affine
        if l == L-1:
            h[l] = softmax
        else:
            h[l] = sigmoid

    U = [np.zeros([Node[l], N]) for l in range(L)] # value after affine
    Z = [np.zeros([Node[l], N]) for l in range(L)] # value after activation


    for m in range(M):
        # initialize error
        dW = {}
        dB = {}
        for l in range(1, L):
            dW[l] = np.zeros([Node[l-1], Node[l]]) # weight
            dB[l] = np.zeros([Node[l], 1]) # bias for affine

        # forward propagation
        U, Z = forward(X_train, Node, W, B, h)
        
        print("U[L-1]:", U[L-1][:5,0])
        print("Z[L-1]:", Z[L-1][:5,0])

        # error of output layer
        delta = [np.zeros([Node[l], N]) for l in range(L)]

        delta[L-1] = Z[L-1] - t_train.reshape([10, N])
        dW[L-1] = np.dot(Z[L-2], delta[L-1].T)
        print("W",W[L-1][:5,0])
        print("dW",dW[L-1][:5,0])
        dB[L-1] = delta[L-1].sum(axis=1).reshape([Node[L-1], 1])

        # back propagation
        for l in range(L-2, 0, -1):
            delta[l] = h_dif(Z[l]) * np.dot(W[l+1], delta[l+1]) # error
            dJ_W = np.dot(Z[l-1], delta[l].T)
            dJ_b = delta[l].sum(axis=1).reshape([Node[l], 1])
            dW[l] = dJ_W
            dB[l] = dJ_b

        for l in range(1, L):
            if(W[l].shape != dW[l].shape):
                print("===== W shapes {}, dW shapes {} do not match".format(W[l].shape, dW[l].shape))
            if(B[l].shape != dB[l].shape):
                print("===== B shapes {}, dB shapes {} do not match".format(B[l].shape, dB[l].shape))
            W[l] = W[l] - e * dW[l] / N
            B[l] = B[l] - e * dB[l] / N

        acc = accuracy(X_test, t_test, Node, W, B, h)
        print("m = {}, acc = {}".format(m, acc))

    print("end")


