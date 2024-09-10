import numpy as np 
from typing import Callable



class Loss: 
    """
    This is base class using to implement loss function
    
    """
    def __init__(
        self, 
        y_pred: np.array, 
        y_true: np.array, 
        function:Callable[[np.ndarray, np.ndarray], float], 
        function_prime: Callable[[np.ndarray, np.ndarray], float]
        ): 
        

        self.y_pred = y_pred
        self.y_true = y_true
        self.function = function 
        self.function_prime = function_prime

    # compute forward loss function 
    def compute(self): 
        return self.function(self.y_pred, self.y_true)
    
    def compute_gradient(self): 
        return self.function_prime(self.y_pred, self.y_true)


class MSELoss(Loss): 
    """ This is implement of mean square error """
    def __init__(
        self, 
        y_pred: np.ndarray, 
        y_true: np.ndarray
        ): 

        self.y_pred = y_pred
        self.y_true = y_true
        self.mse = lambda x, y: np.mean(np.power(x - y, 2))
        self.mse_prime = lambda x, y: 2 * (x - y) / y.size 
        super.__init__(self.y_pred, self.y_true, self.mse, self.mse_prime)


