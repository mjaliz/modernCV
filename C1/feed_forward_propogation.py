import numpy as np

# Activation functions


def sigmoid(x):
    return 1/(1+np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


def relu(x):
    return np.where(x > 0, x, 0)


def linear(x):
    return x


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

# Loss functions


def mse(p, y):
    return np.mean(np.square(p-y))


def mae(p, y):
    return np.mean(np.abs(p-y))


def binary_cross_entropy(p, y):
    return -np.mean(np.sum((y*np.log(p) + (1-y)*np.log(1-p))))


def categorical_cross_entropy(p, y):
    return -np.mean(np.sum(y*np.log(p)))


def feed_froward(inputs, outputs, weights):
    pre_hidden = np.dot(inputs, weights[0]) + weights[1]
    hidden = 1/(1+np.exp(-pre_hidden))
    pred_out = np.dot(hidden, weights[2]) + weights[3]
    mean_squared_error = np.mean(np.square(pred_out - outputs))
    return mean_squared_error
