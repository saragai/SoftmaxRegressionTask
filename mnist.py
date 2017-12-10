import os
import os.path
import pickle
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
    dev = np.sum(np.exp(x), axis=1)
    dev.shape = [dev.size, 1]
    return np.exp(x) / dev

def train(X_train, t_train, w, learning_rate):
    P_train = softmax(np.dot(X_train, w))
    return w - learning_rate * np.dot(X_train.T , P_train - t_train)

def batch_train(X_train, t_train, w, learning_rate, size):
    batch = np.random.randint(60000, size=size)
    X_batch = X_train[batch]
    t_batch = t_train[batch]
    P_batch = softmax(np.dot(X_batch, w))
    return w - learning_rate * np.dot(X_batch.T, P_batch - t_batch)

def accuracy(X_test, t_test, w):
    P_test = (np.dot(X_test, w))
    pred = np.argmax(P_test, axis=1)
    acc = (t_test == pred).sum()/10000
    return acc


if __name__ == "__main__":
    dataset = load()
    t_train = one_of_k(dataset["train_labels"])
    t_test = dataset["test_labels"]
    X_train = dataset["train_images"]/255
    X_test = dataset["test_images"]/255

    epoch = 10
    batch_epoch = 1000
    batch_size = 600
    learning_rate = 0.00001
    batch_learning_rate = 0.001

    fig = plt.figure()
    plt.subplots_adjust(wspace=0.4)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_title("batch")
    ax2.set_title("mini batch")
    ax1.set_xlabel("training data num")
    ax2.set_xlabel("training data num")
    ax1.set_ylabel("accuracy")
    ax2.set_ylabel("accuracy")
    ax1.set_ylim(0,1.0)
    ax2.set_ylim(0,1.0)


    x1 = np.linspace(0, 60000*epoch, num=epoch+1)
    y1 = np.zeros(epoch+1)
    x2 = np.linspace(0, batch_size*batch_epoch, num=batch_epoch+1)
    y2 = np.zeros(batch_epoch+1)

    #w = np.random.rand(784, 10)
    w = np.random.normal(0, 0.1, (784, 10))
    
    w_copy = np.copy(w)
    time1 = time.time()
    y1[0] = accuracy(X_test, t_test, w)
    for i in range(epoch):
        w = train(X_train, t_train, w, learning_rate)
        y1[i+1] = accuracy(X_test, t_test, w)
    print("batch:", accuracy(X_test, t_test, w))
    print("time:", time.time()-time1)

    w = w_copy
    time1 = time.time()
    y2[0] = accuracy(X_test, t_test, w)
    for i in range(batch_epoch):
        w = batch_train(X_train, t_train, w, batch_learning_rate, batch_size)
        y2[i+1] = accuracy(X_test, t_test, w)
    print("mini batch:", accuracy(X_test, t_test, w))
    print("time:", time.time()-time1)

    ax1.plot(x1,y1)
    ax2.plot(x2,y2)
    plt.show()
