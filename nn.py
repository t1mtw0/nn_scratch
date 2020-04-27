import numpy as np
import random


__all__ = ['relu', 'relu_dt', 'sigmoid', 'sigmoid_dt', 'centropy_loss', 'centropy_loss_dt', 'NN']


# Relu

def relu(x):
    return np.maximum(x, 0)

def relu_dt(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x


# Sigmoid

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_dt(x):
    return sigmoid(x) * (1 - sigmoid(x))


# TODO find out how to vectorize batching

# Cross Entropy Loss

def centropy_loss(y, a):
    total = 0
    for i in range(y.shape[0]):
        total += np.sum(-(y[i] * np.log(a[i])))
    return total / y.shape[0]

def centropy_loss_dt(y, a):
    d = np.zeros_like(a)
    for i in range(y.shape[0]):
        d[i] = -(y[i]/ a[i])
    return d


# Neural Network

class NN(object):
    def __init__(self):
        self.weights = [np.random.randn(64, 18) * np.sqrt(1./18), 
                        np.random.randn(18, 2) * np.sqrt(1./2)]
        self.bias = [np.random.randn(18), 
                     np.random.rand(2)]
        
    def feedforward(self, input_batch):
        output = np.dot(input_batch, self.weights[0]) + self.bias[0]
        output = relu(output)
        output = np.dot(output, self.weights[1]) + self.bias[1]
        output = sigmoid(output)
        return output
    
    def backprop(self, input_data):
        # Forward
        target = input_data[1]

        activations = []
        zs = []

        activation = input_data[0]
        activations.append(activation)

        z = np.dot(activation, self.weights[0]) + self.bias[0]
        zs.append(z)

        activation = relu(z)
        activations.append(activation)

        z = np.dot(activation, self.weights[1]) + self.bias[1]
        zs.append(z)

        activation = sigmoid(z)
        activations.append(activation)

        # Backward
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.bias]

        # special case: last layer with loss derivative
        delta = centropy_loss_dt(target, activations[-1]) * sigmoid_dt(zs[-1])

        nabla_w[-1] = np.dot(activations[-2].transpose(), delta)
        nabla_b[-1] = delta

        # loop through the rest
        for i in range(2, len(nabla_w)+1):
            delta = np.dot(delta, self.weights[-i+1].transpose()) * sigmoid_dt(zs[-i])

            nabla_w[-i] = np.dot(activations[-i-1].transpose(), delta)
            nabla_b[-i] = delta

        return (nabla_w, nabla_b)
    
    def update_mini_batch(self, batch, lr):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.bias]

        for i in range(batch[0].shape[0]):
            delta_nabla_w, delta_nabla_b = self.backprop((np.expand_dims(batch[0][i], 0),
                                                          np.expand_dims(batch[1][i], 0)))
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        self.weights = [w - lr * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.bias = [b - lr * nb
                     for b, nb in zip(self.bias, nabla_b)]
        
    def train(self, input_data, num_epochs, lr):
        "Trains the network on input_data for num_epochs times, with learning rate lr"
        for e in range(num_epochs):
            random.shuffle(input_data)
            for data in input_data:
                self.update_mini_batch(data, lr)