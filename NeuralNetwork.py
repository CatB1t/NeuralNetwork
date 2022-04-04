import numpy as np
import math

np.random.seed(0)

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(input_size,))
        self.bias = 0

class Layer:
    def __init__(self, input_size, n_neurons):
        self._neurons = []
        for i in range(n_neurons):
            self._neurons.append( Neuron(input_size) )

    def sigmoid(self, net):
        return 1 / (1 + math.exp(-net))

    def output(self, X):
        output = []
        for neuron in self._neurons:
            sigmoid_vect = np.vectorize(self.sigmoid)
            output.append(sigmoid_vect(np.dot(X, neuron.weights) + neuron.bias))
        return np.array(output).transpose()

class Network:
    def __init__(self):
        self._layers = [] # Array of current neural layers`

    def add_layer(self, number_of_neurons, input_dim):
        self._layers.append(Layer(number_of_neurons, input_dim))

    def predict(self, X):
        last_output = X
        # Simply evaluate network
        for layer in self._layers:
            last_output = layer.forward(last_output)
        # Return the output results
        return last_output

    def fit(self, X, Y):
        # Train the network
        pass

    def _forward_prop(self):
        # forward propagation for network
        pass

    def _backward_prop(self):
        # back propagation of the network
        pass