import numpy as np 
from .layer import * 
from .loss import Loss, MSELoss

from typing import List


class Network: 
    """
    Class nay tong hop cac thanh phan duoc tao tu truoc do de tao thanh mang neural hoan chinh 
    """

    def __init__(self):

        # danh sach layer cua lop 
        self.layers = [] 
        # ham loss 
        self.loss = None 
    

    # add layer into network 
    def add(self, layer: Layer): 
        self.layers.append(layer)


    # set loss
    def setLoss(self, loss: Loss): 
        self.loss = loss


    # predict 
    def predict(self, X: np.ndarray):
        num_samples = X.shape[0]
        res = []
        for i in range(num_samples): 
            inst_i = X[i]
            for layer in self.layers: 
                inst_i = layer.forward(inst_i)

            res.append(inst_i)

        return np.array(res)



    # train 
    def train(self,X_train: np.ndarray, y_train: np.ndarray,  learning_rate: float, epochs: int): 
        
        samples = X_train.shape[0]

        # training loop 
        for i in range(epochs): 
            err = 0 

            for j in range(samples): 
                output = X_train[j]
                # forward 
                for layer in self.layers: 
                    output = layer.forward(output)
                
                # compute loss 
                err += self.loss.forward(output, y_train[j])

                # backward 
                err = self.loss.backward(output, y_train[i])
                for layer in reversed(self.layers): 

                    err = layer.backward(err, learning_rate)

            # the average error on all sample 
            err /= samples 
            print(f'Epoch {i}: error = {err}')


