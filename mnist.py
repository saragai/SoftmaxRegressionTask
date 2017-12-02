import os
import os.path
import pickle
import numpy as np
from load import load


def one_of_k(data):
    labels = np.zeros((data.size, 10))
    for arg, label in enumerate(labels):
        label[data[arg]] = 1
    return labels

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

dataset = load()
train_labels = one_of_k(dataset["train_labels"])
test_labels = one_of_k(dataset["test_labels"])
train_images = dataset["train_images"]/255
test_images = dataset["test_images"]/255

#w = np.random.rand(784, 10)
w = np.random.normal(0, 0.1, (784, 10))
learning_rate = 0.01


for x, t in zip(train_images, train_labels):
    x.shape = [784, 1]
    t.shape = [1, 10]
    p = softmax(np.dot(w.T, x))
    w -= learning_rate * np.dot(x, p.T - t)

acc = 0
for x, t in zip(test_images, test_labels):
    p = np.argmax(softmax(np.dot(w.T, x)))
    if t[p]:
        acc += 1

print(acc)

