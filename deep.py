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
    dev = np.sum(np.exp(x), axis=1)
    print("dev",dev.shape)
    dev.shape = [dev.size, 1]
    return np.exp(x) / dev

    
def ReLu(x):
    mask = x < 0
    dx = x[mask] = 0
    return dx

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_dif(z):
    return z * (1-z)


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
    #"""
    #Hyper Param
    M = 1 # maximum iterations (epoch)
    e = 0.001 # learning rate

    Node = [784, 100, 10]
    L = len(Node)

    h = sigmoid
    h_dif = sigmoid_dif

    # initialize
    W = {}
    B = {}
    for l in range(1, L):
        W[l] = np.zeros([Node[l-1], Node[l]]) #weight
        B[l] = np.zeros([Node[l], 1]) # bias for affine

    Z = [np.zeros([Node[l], 1]) for l in range(L)] # value after affine
    U = [np.zeros([Node[l], 1]) for l in range(L)] # value after activation


    for m in range(M):
        # initialize error
        dW = {}
        dB = {}
        for l in range(1, L):
            dW[l] = np.zeros([Node[l-1], Node[l]]) #weight
            dB[l] = np.zeros([Node[l], 1])# bias for affine

        # iterate for data
        for n in range(N):
            # forward propagation
            Z[0] = X_train[n].reshape([Node[0], 1])
            for l in range(1, L):
                U[l] = np.dot(W[l].T, Z[l-1]).reshape([Node[l],1]) + B[l]
                #print("W[{}].T.shape : {}".format(l, W[l].T.shape))
                #print("Z[{}].shape : {}".format(l, Z[l].shape))
                #print("U[{}].shape : {}".format(l, U[l].shape))
                Z[l] = h(U[l])

            # error of output layer
            delta = [np.zeros([Node[l], 1]) for l in range(L)]

            delta[L-1] = Z[L-1] - t_train[n].reshape([10,1])
            dW[L-1] = dW[L-1] + np.dot(Z[L-2], delta[L-1].T)
            dB[L-1] = dB[L-1] + delta[L-1]

            # back propagation
            for l in range(L-2, 0, -1):
                delta[l] = h_dif(Z[l]) * np.dot(W[l+1], delta[l+1]) # error
                #print("U[{}].shape : {}".format(l, U[l].shape))
                #print("h(U[{}]).shape :{}".format(l, h(U[l]).shape))
                #print("W[{}] * delta[{}] .shape :{}".format(l+1,l+1, np.dot(W[l+1], delta[l+1]).shape))
                dJ_W = np.dot(Z[l-1], delta[l].T)
                dJ_b = delta[l]
                dW[l] = dW[l] + dJ_W
                dB[l] = dB[l] + dJ_b

        for l in range(1, L):
            W[l] = W[l] - e * dW[l] / N
            B[l] = B[l] - e * dB[l] / N

        #"""
    print("end")
