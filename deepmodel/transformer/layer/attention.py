import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 


class MultiHeadAttention(nn.Module): 

    """
    Class thực hiện cơ chế attention
    Input : 
        - d_model : số lượng dimension của embedding model 
        - num_head : số lượng head của attention
    """

    def __init__(self, d_model: int, num_head: int):
        super(MultiHeadAttention, self).__init__()

        # hyper pagrams setup 
        self.d_model = d_model
        self.num_head = num_head

        # setup weight 
        self.w_key = torch.zeros(size = (self.d_model, self.d_model))
        self.w_query = torch.zeros(size = (self.d_model, self.d_model))
        self.w_value = torch.zeros(size = (self.d_model, self.d_model))


    # forward function 
    def forward(self): 
        pass 