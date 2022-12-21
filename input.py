import numpy as np

class input:
    def __init__(self):
        self.inputVal = 0

    # I called this a sigmoid function because this allows the neural network to call the function to get the input value
    def sigmoid_function(self):
        return self.inputVal