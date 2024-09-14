import numpy as np 
from scipy import signal

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
    This is fully-connected layer implementation 
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
        return self.output

    def backward(self, output_error: np.array, learning_rate: np.array):  
        """Backward propagation, return de/dx"""
        input_error = np.dot(output_error, self.weights.T) 
        weight_gradient = np.dot(self.input.T, output_error)   
        self.weights -= learning_rate * weight_gradient
        return input_error


class Conv(Layer): 

    """
    This is convolution layer implementation 
    
    """


    def __init__(self, input_shape: tuple, kernel_size: int, depth: int): 
        """
        depth : is the number of kernel 
        input_shape : is the shape of input tensor 
        kernel_size : is the shape of kernel matrix 
        """

        input_depth, input_height, input_width = input_shape[0], input_shape[1], input_shape[2]
        self.input_shape = input_shape 
        self.input_depth = input_depth 
        
        self.depth = depth 
        # setup output shape 
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        # setup kernel shape 
        self.kernel_shape = (depth, input_depth, kernel_size, kernel_size)

        # init kernel 
        self.kernels = np.random.randn(*self.kernel_shape)
        # init bias
        self.bias = np.random.randn(*self.output_shape)


    # forward function 
    def forward(self, input): 
        self.input = input
        # set ouput as bias  
        self.output = np.copy(self.bias)

        # loop all kernels 
        for i in range(self.depth): 
            # loop  
            for j in range(self.input_depth): 
                
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output 
    
    # backward
    def backward(self, output_gradient: np.ndarray, learning_rate: float): 

        # compute kernel gradient dE/dk 
        kernel_gradient = np.zeros(self.kernel_shape)

        # compute input_gradient 
        input_gradient = np.zeros(self.input_shape)


        for i in range(self.depth): 

            for j in range(self.input_depth): 

                kernel_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[i] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        # update
        self.kernels -= learning_rate * kernel_gradient

        # dE/dBi = dE/dYi 
        self.bias -= learning_rate * output_gradient
        return input_gradient


    

if __name__ == '__main__': 

    pass 


    # Linear testing 
    def linear_forward_testing(): 
        pass 


    def  linear_backward_testing(): 
        pass 
    


    # convolution testing 
    def conv_testing(): 
        conv = Conv((3, 3,3), 2, 2)


        input = np.random.randn(3,3,3)

        # forward testing     
        print(conv.forward(input))

        # backward testing 
        print(conv.backward(np.random.randn(*conv.output_shape), 0.1))



    conv_testing()
    
        





