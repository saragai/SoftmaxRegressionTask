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
    #print("softmax shape", dev.shape)
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

def forward_test():
    X = np.array([[0, 1, 2],
                  [3, 4, 5]])
    Node = [2, 1, 3]
    W = {1: np.array([[1],
                     [2]]),
         2: np.array([[0, 1, 2]])}
    B = {1: np.array([[1]]),
         2: np.array([[0],
                     [1],
                     [2]])}
    h = {1: lambda x: x,
         2: lambda x: x}
    U, Z = forward(X, Node, W, B, h)
    if (Z[2] == np.array([[0,0,0],[8,11,14],[16,22,28]])).all():
        pass
    else:
        print("=== error forward test ===\n{}\n".format(Z))
forward_test()

def accuracy(X, t, Node, W, B, h):
    _, Z = forward(X, Node, W, B, h)
    L = len(Node)
    arg = np.argmax(Z[L-1], axis=0)
    return np.average(arg == t)


if __name__ == "__main__":
    dataset = load.load()
    t_train = one_of_k(dataset["train_labels"]).T
    t_test = dataset["test_labels"]
    X_train = dataset["train_images"].T/255
    X_test = dataset["test_images"].T/255

    print("t_train.shape", t_train.shape)
    print("X_train.shape", X_train.shape)
    N = 1000 # Number of data

    #Hyper Param
    M = 2000 # maximum iterations (epoch)
    e = 0.1 # learning rate
    Node = [784, 256, 128, 64, 10]
    L = len(Node)

    # for Plot
    fig = plt.figure()
    plt.subplots_adjust(wspace=0.4)
    ax = fig.add_subplot(111)
    ax.set_title("Feed Forward Neural Network (noise in train & test data)")
    ax.set_xlabel("epoch (1 epoch has {} data)".format(N))
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.0)
    xplot = [[] for i in range(6)]
    yplot = [[] for i in range(6)]

    W_init = {}
    for l in range(1, L):
        W_init[l] = np.random.normal(0, 0.1, (Node[l-1], Node[l])) #weight

    #random
    #"""
    X_rand = [X_test.copy() for i in range(6)]
    for i in range(6):
        for n in range(10000):
            num_rand = np.random.binomial(n=784, p=i*0.05)
            rand_mask = np.random.choice(784, num_rand, replace=False)
            X_rand[i][rand_mask, n] = np.random.rand(num_rand)
    #"""
    # =================
    #    training
    # =================
    for noise in range(6):

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
            # batch
            #X = X_train
            #t = t_train

            # mini batch  Ver.random
            batch_mask = np.random.randint(0, 60000, N)
            X = X_train[:, batch_mask]
            t = t_train[:, batch_mask]

            num_rand = np.random.binomial(n=784*N, p=noise*0.05)
            rand_mask_arg = np.random.choice(784*N, num_rand, replace=False)
            rand_mask = np.zeros(784*N).astype(np.bool_)
            rand_mask[rand_mask_arg] = True
            rand_mask.shape = [784, N]
            X[rand_mask] = np.random.rand(num_rand)
            
            # initialize error
            delta = [np.zeros([Node[l], N]) for l in range(L)]
            dW = {}
            dB = {}
            for l in range(1, L):
                dW[l] = np.zeros_like(W[l]) # weight
                dB[l] = np.zeros_like(B[l]) # bias for affine

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
                if(W[l].shape != dW[l].shape):
                    print("===== W shapes {}, dW shapes {} do not match".format(W[l].shape, dW[l].shape))
                if(B[l].shape != dB[l].shape):
                    print("===== B shapes {}, dB shapes {} do not match".format(B[l].shape, dB[l].shape))
                W[l] = W[l] - e * dW[l] / N
                B[l] = B[l] - e * dB[l] / N


            if m % 100 == 0:
                xplot[noise].append(m)
                #acc = accuracy(X_test, t_test, Node, W, B, h)
                acc = accuracy(X_rand[noise], t_test, Node, W, B, h)
                yplot[noise].append(acc)
                print("m = {}, rand = {}%, acc = {}".format(m, noise*5, acc))
       print("")

    if True:
        for i in range(6):
            ax.plot(xplot[i], yplot[i], label="{} %".format(i*5))
        ax.legend()
        plt.show()
    print("end")


