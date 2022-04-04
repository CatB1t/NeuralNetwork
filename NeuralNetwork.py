import numpy as np
import math

np.random.seed(0)


class Layer:
    def __init__(self, input_size, n_neurons):
        self._weights = [] # List of weights for each neuron
        self.input = []
        self.output = []
        for i in range(n_neurons):
            self._weights.append( np.random.uniform(low=-0.5, high=0.5, size=(input_size,)) )

    def sigmoid(self, net):
        return 1 / (1 + math.exp(-net))

    def forward(self, x):
        self.input = x
        output = []
        for neuron in self._weights:
            sigmoid_vect = np.vectorize(self.sigmoid)
            output.append(sigmoid_vect(np.dot(x, neuron)))
        self.output = np.array(output).transpose()
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights)
        weights_error = np.dot(self.input, output_error)
        # update parameters
        self.weights -= learning_rate * weights_error
        return input_error

class Network:
    def __init__(self):
        self._layers = [] # Array of current neural layers`

    def add_layer(self, layer):
        self._layers.append(layer)

    def predict(self, X):
        return self._forward_prop(X)

    def fit(self, X, Y, epochs=20, learning_rate=0.1):
        # Should first do forward propagation
        for i in range(epochs):
            # For each sample
            for j in range(len(X)):
                y_pred = self._forward_prop(X[j])
                error = self._error(Y[j], y_pred)
                #print("Iteration: {}, Sample: {},  Error: {}".format(i+1, j+1, error))
                self._backward_prop(error)

    def _error(self, y, y_pred):
        return np.sum((y - y_pred) ** 2)

    def _forward_prop(self, x):
        # forward propagation for network
        last_output = x
        # Simply evaluate network
        for layer in self._layers:
            last_output = layer.forward(last_output)
        # Return the output results
        return last_output

    def _backward_prop(self, error):
        learning_rate = 0.1
        # back propagation of the network
        # Update the weights at the output layer
        # Then go for each layer except the output and the input layer and update their weights
        ## hardcoded for input(2) , hidden_layer(3), output(1)
