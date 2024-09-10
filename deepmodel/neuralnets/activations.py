import numpy as np 

class Activations: 
    """
    Base Activation layers implementation 
    """
    def __init__(self, activation, activation_prime): 
        self.activation = activation
        self.activation_prime = activation_prime


    def forward(self, input: np.array):
        """ Forward propagation """
        self.input = input
        return self.activation(self.input)


    def backward(self, output_gradient, learning_rate): 
        """ Backward propagation """
        return np.multiply(output_gradient, self.activation_prime(self.input))



class Tanh(Activations): 

    def __init__(self): 
        self.tanh = lambda x: np.tanh(x)
        self.tanh_prime = lambda x: 1 - np.tanh(x) **2 
        super().__init__(self.tanh, self.tanh_prime) 


class Sigmoid(Activations): 

    def __init__(self): 
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.sigmoid_prime = lambda x: self.sigmoid(x) * (1 -  self.sigmoid(x))