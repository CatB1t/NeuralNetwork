import numpy as np


class Sigmoid:
    def __init__(self):
        self.input = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_prime(self, x):
        s = self._sigmoid(x)
        return s * (1 - s)

    def forward(self, x):
        self.input = x
        return self._sigmoid(x)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self._sigmoid_prime(self.input))


class Layer:
    def __init__(self, input_size, n_neurons):
        self.weights = np.random.randn(n_neurons, input_size)
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        self.output = np.dot(self.weights, x)
        return self.output

    def backward(self, last_layer_d, learning_rate):
        gd_weights = -np.dot(last_layer_d, self.input.T)
        self.weights += learning_rate * gd_weights
        d_input = np.dot(self.weights.T, last_layer_d)
        return d_input


class Network:
    def __init__(self):
        self._layers = []

    def add_layer(self, layer, activation):
        self._layers.append(layer)
        self._layers.append(activation)

    def fit(self, x_train, y_train, epochs=100, learning_rate=0.01):
        for i in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                y_pred = self.forward(x)
                error += self._error(y, y_pred)
                self.backward(y, y_pred, learning_rate)

            print("iteration: {}, Error: {}".format(i+1, error / len(x_train)))

    def argmax(self): # TODO for predicting final class
        pass

    def predict(self, X):
        # TODO use max values to predict class
        output = []
        for x in X:
            l_out = self.forward(x)
            output.append(l_out)
        return output

    def _error(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def forward(self, x):
        # forward propagation for network
        last_output = x
        # Simply evaluate network
        for layer in self._layers:
            last_output = layer.forward(last_output)
        # Return the output results
        return last_output

    def backward(self, y, y_pred, learning_rate):
        last_layer_ed = np.mean((y_pred - y))
        for layer in reversed(self._layers):
            last_layer_ed = layer.backward(last_layer_ed, learning_rate)