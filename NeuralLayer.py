import Neuron


class NeuralLayer:
    def __init__(self, n, input_size):
        self._neurons = []
        for i in range(n):
            self._neurons.append(Neuron(input_size))

    def evaluate_layer(self):
        # Should return a vector of size n
        pass