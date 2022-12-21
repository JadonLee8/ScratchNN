import Sigmoids
import numpy as np
from input import input

class network:
    # precondition: layers is an array integers representing the number of nodes in each layer
    # precondition: inputs is an integer representing the number of inputs
    def __init__(self, layers, inputs):
        self.weights = []
        self.biases = []
        self.inputs = []
        self.layers = layers

        # create input nodes
        for i in range(0, inputs):
            self.inputs.append(input())
        self.nodes = []
        self.nodes.append(self.inputs)

        #generate random weights and biases to start
        self.weights.append([])
        self.biases.append([])
        for i in range(0, layers[0]):
            self.weights[0].append(np.random.random(size=len(self.inputs))*10 - 5)
            self.biases[0].append(np.random.random()*10 - 5)
        for i in range(1, len(layers)):
            self.weights.append([])
            self.biases.append([])
            for j in range(0, layers[i]):
                # may be a problem here
                self.weights[i].append(np.random.random(size=self.layers[i-1])*10 - 5)
                self.biases[i].append(np.random.random()*10 - 5)


        # create the neurons
        for i in range(1, len(layers)):
            self.nodes.append([])
            for j in range(0, layers[i]):
                # could replace first param with the layer number and can access the layer with self.nodes[i] later
                self.nodes[i].append(Sigmoids.sigmoid(self.nodes[i-1], self.weights[i - 1][j], self.biases[i - 1][j]))


    # precondition: inputs is an array of inputs whose length is the number of inputs
    def feed_forward(self, inputs):
        for i in range(0, len(self.inputs)):
            self.inputs[i].inputVal = inputs[i]
        for sigmoid in self.nodes[len(self.layers)-1]:
            print(sigmoid.sigmoid_function())