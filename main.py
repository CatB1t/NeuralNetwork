import numpy as np
from NeuralNetwork import Network, Layer, Sigmoid


np.random.seed(0)

# XOR problem
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = Network()
network.add_layer(Layer(2, 3), activation=Sigmoid()) # Hidden layer
network.add_layer(Layer(3, 1), activation=Sigmoid()) # Output layer
network.fit(X, y, epochs=2000, learning_rate=0.6)

X_test = [[0, 0], [0, 1], [1, 0], [1, 1]]
print(network.predict(X_test))
