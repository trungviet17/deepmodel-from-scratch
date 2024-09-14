import numpy as np 
from typing import Callable



class Loss: 
    """
    This is base class using to implement loss function
    
    """
    def __init__(
        self, 
        function:Callable[[np.ndarray, np.ndarray], float], 
        function_prime: Callable[[np.ndarray, np.ndarray], float]
        ): 
        
        self.function = function 
        self.function_prime = function_prime

    # compute forward loss function 
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray): 
        return self.function(y_pred, y_true)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray): 
        return self.function_prime(y_pred, y_true)


class MSELoss(Loss): 
    """ This is implement of mean square error """
    def __init__(self): 

        
        self.mse = lambda x, y: np.mean(np.power(x - y, 2))
        self.mse_prime = lambda x, y: 2 * (x - y) / y.size 
        super().__init__(self.mse, self.mse_prime)


class BCELoss(Loss): 

    def __init__(self): 
        # x is y_pred, y is y_true
        self.bce = lambda x, y: -np.mean(y * np.log(x) + (1 - x) * np.log(1 - y))
        self.bce_prime = lambda x, y: ( (1 - y)/(1 - x) - y / x) / np.size(y)
        super().__init__(self.bce, self.bce_prime)