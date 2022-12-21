
import numpy as np

class sigmoid:
    def __init__(self, layer, weights, bias):
        self.layer = layer
        self.weights = weights
        self.bias = bias
        self.inputs = []
        for i in range(0, len(layer)):
            self.inputs.append(0)
        self.last_output = -1

    def sigmoid_function(self):
        # compute inputs
        for i in range(0, len(self.layer)):
            self.inputs[i] = self.layer[i].sigmoid_function()
        # compute sigmoid input
        sigmoid_input = np.sum(np.matmul(self.inputs, self.weights)) + self.bias
        # compute sigmoid output
        self.last_output = 1/(1+np.exp(-sigmoid_input))
        return self.last_output