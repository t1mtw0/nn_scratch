import numpy as np
from nn import *

# Sample data
input_data = [(np.random.randn(8, 64), np.array([1, 0, 1, 0, 0, 1, 1, 0])), 
              (np.random.randn(8, 64), np.array([0, 1, 1, 1, 1, 1, 0, 1])), 
              (np.random.randn(8, 64), np.array([0, 0, 0, 0, 0, 1, 1, 0])),
              (np.random.randn(8, 64), np.array([1, 0, 0, 1, 0, 0, 0, 0])),
              (np.random.randn(8, 64), np.array([0, 1, 1, 1, 1, 1, 1, 0])),
              (np.random.randn(8, 64), np.array([1, 1, 1, 0, 1, 0, 1, 1]))]

network = NN()

outputs = network.feedforward(input_data[0][0])
print("Loss before: ", centropy_loss(input_data[0][1], outputs))

# TODO add support for command line arguments for num_epochs and lr

# Training
network.train(input_data, num_epochs=10, lr=0.01)

outputs = network.feedforward(input_data[0][0])
print("Loss after: ", centropy_loss(input_data[0][1], outputs))