import numpy as np
import matplotlib.pyplot as plt
from load import load
import time

def mode(label, dist):
    x = np.zeros(10)
    d = np.zeros(10)
    for arg, l in enumerate(label):
        x[l] += 1
        d[l] += dist[arg]
    arg = (x == np.max(x))
    if np.sum(arg) == 1:
        return np.argmax(arg)
    else:
        d[~arg] = np.inf
        return np.argmin(d)

def modetest():
    label = [3,1,2,2,1]
    dist = [0.1, 0.3, 0.2, 0.2, 0.4]
    arg = mode(label, dist)
    if arg == 2:
        print("mode works")
    else:
        print("mode does not work. arg:", arg)

modetest()

if __name__ == "__main__":
    dataset = load.load()
    t_train = dataset["train_labels"]
    t_test = dataset["test_labels"]
    X_train = dataset["train_images"]/255
    X_test = dataset["test_images"]/255

    """
    X_train[X_train > 0] = 1
    X_test[X_test > 0] = 1
    for i in range(28):
        print(X_train[0][i*28:(i+1)*28])
    #"""

    N = 1000
    K = [1,2,3,4,5,6,7,8,9]
    accuracy = []
    print("X_train", X_train.shape)
    for k in K:
        acc = 0
        for n in range(N):
            delta = X_test[n].reshape(1, 784) - X_train
            distance = np.sum(delta ** 2, axis=1)
            neararg = np.argsort(distance)[:k]
            ans = mode(t_train[neararg], distance[neararg])

            if ans == t_test[n]:
                acc += 1
            else:
                print("answer {}, pred {}".format(t_test[n], ans))
        print("k = {}, acc = {}".format(k, acc/N))
        accuracy.append(acc/N)
    plt.plot(K, accuracy)
    plt.title("kNN")
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.ylim(0,1.0)
    plt.show()
            
