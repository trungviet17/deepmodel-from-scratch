import numpy as np 


# Base layer class 
class Layer: 

    def __init__(self): 
        self.input = None
        self.output = None  


    # forward function 
    def forward(self, input: np.array): 
        raise NotImplementedError
    
    # 
    def backward(self, output_error: np.array, learning_rate: float): 
        raise NotImplementedError


class Linear(Layer): 
    """
    This is fully-connected layer implement 
    """
    
    def __init__(self, input_size: int, output_size: int): 
        """ 
        - input size is the number of input neurons 
        - output size is the number of output neurons 
        """ 
        self.weights = np.random.rand(input_size, output_size) - 0.5 
        self.bias = np.random.rand(1, output_size) - 0.5 

    def forward(self, input: np.array): 
        """Forward propagation, return output of given input"""
        self.input = input 
        self.output = np.dot(self.input, self.weights) + self.bias 


    def backward(self, output_error: np.array, learning_rate: np.array): 
        """Backward propagation, return de/dx"""
        input_error = np.dot(output_error, self.input.T)
        weight_gradient = np.dot(self.weights.T, output_error)
        self.weights -= learning_rate * weight_gradient
        # the derative of bias is the same as ouput_error 
        self.bias -= learning_rate * output_error
        return input_error  
    


    
        





