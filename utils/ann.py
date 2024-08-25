import numpy as np
import utils.activation_function as af


class Neuron:
    def __init__(self, weights, bias):
        """Initializes a Neuron with given weights and bias."""
        self.weights = np.array(weights)
        self.bias = bias

    def forward(self, inputs):
        """Calculates the output of the neuron for given inputs."""
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return af.relu(weighted_sum)


class Layer:
    def __init__(self, neurons):
        """Initializes a Layer with a list of neurons."""
        self.neurons = neurons

    def forward(self, inputs):
        """Calculates the output of the layer by passing inputs through each neuron."""
        return [neuron.forward(inputs) for neuron in self.neurons]


class NeuralNetwork:
    def __init__(self, layers):
        """Initializes a Neural Network with a list of layers."""
        self.layers = layers

    def forward(self, inputs):
        """Calculates the output of the neural network by passing inputs through each layer."""
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
