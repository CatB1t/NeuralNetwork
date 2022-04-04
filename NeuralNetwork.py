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

    def add_layer(self, layer):
        self._layers.append(layer)

    def predict(self, X):
        return self._forward_prop(X)

    def fit(self, X, Y):
        # Should first do forward propagation
        n_iteration = 20
        for i in range(n_iteration):
            y_pred = self._forward_prop(X)
            # Then calculate error
            error = self._error(Y, y_pred)
            print("Iteration {} Error: {}".format(i, error))
            # do backward propagation
            self._backward_prop(error)

    def _error(self, Y, y_pred):
        return np.sum((Y - y_pred) ** 2)

    def _forward_prop(self, X):
        # forward propagation for network
        last_output = X
        # Simply evaluate network
        for layer in self._layers:
            last_output = layer.output(last_output)
        # Return the output results
        return last_output

    def _backward_prop(self, error):
        learning_rate = 0.1
        # back propagation of the network
        # Update the weights at the output layer
        # Then go for each layer except the output and the input layer and update their weights

        pass