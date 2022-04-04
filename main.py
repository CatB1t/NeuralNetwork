import numpy as np
from NeuralNetwork import Network, Layer

# XOR problem
X = [[0,0], [0,1], [1,0], [1,1]]
y = [[1], [0], [0], [1]]

network = Network()
network.add_layer(Layer(2, 3))
network.add_layer(Layer(3, 1))
network.fit(X, y, epochs=2,learning_rate=0.1)

print(network.predict(X))
