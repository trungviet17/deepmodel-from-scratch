import torch 
import torch.nn as nn 
import math

class PositionalEncoding(nn.Module): 
    """
    Đây là lớp positional encoding (mã hóa vị trí trong câu của từng token)
    Input : 
        1. d_model : là kích thước của dimmension 
        2. max_seq_len : chiều dài của toàn bộ câu văn bản 
    
    """
    def __init__(self, d_model : int, max_seq_len : int):

        self.d_model = d_model 
        self.max_seq_len = max_seq_len

        # positional encoding 
        pe = torch.zeros(size =(self.max_seq_len, self.d_model))
        # position 
        position = torch.arange(0, self.max_seq_len, dtype = torch.float)

        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype = torch.float) * -(math.log(10000))
        )
        # compute pe for each 2i and 2i + 1 
        pe[:, 0::2] = torch.sin(position * div_term)    
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x): 
        x = x + self.pe[:, :x.size(1)]
        return x