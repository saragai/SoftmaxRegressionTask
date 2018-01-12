import numpy as np
import matplotlib.pyplot as plt
from load import load
import time

def one_of_k(data):
    labels = np.zeros((data.size, 10))
    for arg, label in enumerate(labels):
        label[data[arg]] = 1
    return labels

def softmax(x):
    dev = np.sum(np.exp(x), axis=0)
    dev.shape = [1, dev.size]
    return np.exp(x) / dev
    
def ReLU(x):
    mask = x < 0
    x[mask] = 0
    return x

def ReLU_dif(x):
    mask = x < 0
    x[mask] = 0
    x[~mask] = 1
    return x

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def sigmoid_dif(u):
    z = sigmoid(u)
    return z * (1 - z)

def forward(X, Node, W, B, h):
    N = len(X[0])
    L = len(Node)
    Z = [np.zeros([Node[l], N]) for l in range(L)] # value after affine
    U = [np.zeros([Node[l], N]) for l in range(L)] # value after activation
    Z[0] = X
    for l in range(1, L):
        U[l] = np.dot(W[l].T, Z[l-1]) + B[l]
        Z[l] = h[l](U[l])
    return U, Z

def accuracy(X, t, Node, W, B, h):
    _, Z = forward(X, Node, W, B, h)
    L = len(Node)
    arg = np.argmax(Z[L-1], axis=0)
    return np.average(arg == t)

def noise_on_train(X):
    num_rand = np.random.binomial(n=784*N, p=0.15)
    rand_mask_arg = np.random.choice(784*N, num_rand, replace=False)
    rand_mask = np.zeros(784*N).astype(np.bool_)
    rand_mask[rand_mask_arg] = True
    rand_mask.shape = [784, N]
    X[rand_mask] = np.random.rand(num_rand)
    return X

if __name__ == "__main__":
    dataset = load.load()
    t_train = one_of_k(dataset["train_labels"]).T #shape (10, 60000)
    t_test = dataset["test_labels"]
    X_train = dataset["train_images"].T/255 #shape (784, 60000)
    X_test = dataset["test_images"].T/255 #shape (784, 10000)


    #Hyper Param
    N = 1000 # number of minibatch 
    M = 10000 # maximum iterations (epoch)
    e = 0.1 # learning rate
    w_init_mean = 0
    w_init_sigma = 0.1
    NodeL = [[784, 512, 256, 128, 64, 10],
            [784, 256, 128, 64, 10],
            [784, 256, 64, 10],
            [784, 256, 10]]
    LL = [6,5,4,3]

    

    # noise
    #"""
    noise_on_train_flag = True
    noise_on_test_flag = True
    noise_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

    nl = len(noise_list)
    #"""

    # for Plot
    fig = plt.figure()
    plt.subplots_adjust(wspace=0.4)
    ax = fig.add_subplot(111)
    ax.set_title("Feed Forward Neural Network (noise in train & test data)")#(noise in train & test data)")
    ax.set_xlabel("epoch (1 epoch has {} data)".format(N))
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.0)
    xplot = [[] for i in range(len(LL))]
    yplot = [[] for i in range(len(LL))]

    acc_interval = 100

    W_init = {}

    # add noise on test data
    #"""
    X_rand = X_test.copy()
    for n in range(10000):
        num_rand = np.random.binomial(n=784, p=0.15)
        rand_mask = np.random.choice(784, num_rand, replace=False)
        X_rand[rand_mask, n] = np.random.rand(num_rand)
    #"""
    # =================
    #    training
    # =================
    for i in range(4):
        L = LL[i]
        Node = NodeL[i]
        for l in range(1, L):
            W_init[l] = np.random.normal(w_init_mean, w_init_sigma, (Node[l-1], Node[l])) #weight

        # =================
        #    initialize
        # =================
        W = {}
        B = {}
        U = {}
        Z = {}
        h = {}
        h_dif = {}
        for l in range(1, L):
            W[l] = W_init[l].copy()
            B[l] = np.zeros([Node[l], 1]) # bias for affine
            U[l] = np.zeros([Node[l], N]) # value in layer
            Z[l] = np.zeros([Node[l], N]) # value after layer
            if l == L-1:
                h[l] = softmax
            elif l == 1:
                h[l] = ReLU
                h_dif[l] = ReLU_dif
            else:
                h[l] = sigmoid #ReLU
                h_dif[l] = sigmoid_dif #ReLU_dif
 
        for m in range(M+1):
             # mini batch  Ver.random
             batch_mask = np.random.randint(0, 60000, N)
             X = X_train[:, batch_mask]
             t = t_train[:, batch_mask]
             
             # noise on training data
             if noise_on_train_flag:
                 X = noise_on_train(X)
             
             # initialize error
             delta = {}#[np.zeros([Node[l], N]) for l in range(L)]
             dW = {}
             dB = {}
             for l in range(1, L):
                 dW[l] = np.zeros_like(W[l]) # weight
                 dB[l] = np.zeros_like(B[l]) # bias for affine
                 delta[l] = np.zeros([Node[l], N])
 
             # forward propagation
             U, Z = forward(X, Node, W, B, h)
 
             # error of output layer
             delta[L-1] = Z[L-1] - t
             dW[L-1] = np.dot(Z[L-2], delta[L-1].T)
             dB[L-1] = delta[L-1].sum(axis=1).reshape([Node[L-1], 1])
 
             # back propagation
             for l in range(L-2, 0, -1):
                 delta[l] = h_dif[l](U[l]) * np.dot(W[l+1], delta[l+1]) # error
                 dW[l] = np.dot(Z[l-1], delta[l].T)
                 dB[l] = delta[l].sum(axis=1).reshape([Node[l], 1])
 
             for l in range(1, L):
                 W[l] = W[l] - e * dW[l] / N
                 B[l] = B[l] - e * dB[l] / N
 
             # accuracy
             if m % acc_interval == 0:
                 xplot[i].append(m)
                 if noise_on_test_flag:
                     acc = accuracy(X_rand, t_test, Node, W, B, h)
                 else:
                     acc = accuracy(X_test, t_test, Node, W, B, h)
 
                 yplot[i].append(acc)
                 print("m = {}, Node = {}%, acc = {}".format(m, LL[i], acc))
        print("")
 
    if True:
        for i,L in enumerate(LL):
            ax.plot(xplot[i], yplot[i], label="{} layers".format(L))
        ax.legend()
        plt.show()
