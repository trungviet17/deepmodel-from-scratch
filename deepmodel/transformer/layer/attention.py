import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 
import math 

class MultiHeadAttention(nn.Module): 

    """
    Class thực hiện cơ chế attention, embedding vec được chia nhỏ để học qua nhiều num_head -> có nhiều góc nhìn hơn về dữ liệu. 
    Input được huấn luyện qua 3 layer tương ứng với từng giá trị key, value, query 
    
    Input : 
        - d_model : số lượng dimension của embedding model 
        - num_head : số lượng head của attention
    """

    def __init__(self, d_model: int, num_head: int):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_head == 0 

        # hyper pagrams setup 
        self.d_model = d_model
        self.num_head = num_head
        self.d_tensor = self.d_model // self.num_head

        # setup weight 
        self.key_layer = nn.Linear(self.d_model, self.d_model)
        self.value_layer = nn.Linear(self.d_model, self.d_model)
        self.query_layer = nn.Linear(self.d_model, self.d_model)
        
        # setup output layer
        self.output_layer = nn.Linear(self.d_model, self.d_model)

    # split embedding vector to each head 
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_head,d_model // self.num_head ).transpose(1, 2)

    # inverse function of split_heads 
    def concat(self, x):
        batch_size, head, length, d_tensor = x.size()
        d_model = head * d_tensor 

        return x.transpose(1, 2).contiguous().view(batch_size, length, d_model)


    # compute attention score (3.2.1 in paper)
    def compute_attention(self, query, key, value,  mask = None): 
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_tensor) 

        if mask is not None: 
            scores = scores.masked_fill(mask == 0, -1e9)
        
        scores = torch.softmax(scores, dim = -1)
        return torch.matmul(scores, value)
    
 
    # forward function 
    def forward(self, x): 
        query = self.split_heads(self.query_layer(x))
        value = self.split_heads(self.value_layer(x))
        key = self.split_heads(self.key_layer(x))

        att_scores = self.compute_attention(query, key, value)
        output = self.output_layer(self.concat(att_scores))

        return output 