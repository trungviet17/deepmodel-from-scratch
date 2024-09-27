import torch 
import torch.nn as nn 
import math 
from layer.attention import MultiHeadAttention
from layer.pointwise_feed_forward import Position_wise_feed_forward


class EncoderLayer(nn.Module): 

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.multi_att = MultiHeadAttention(d_model, num_heads)
        self.feedforward = Position_wise_feed_forward(d_model, d_ff)

        # setup norm and dropout 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)



    def forward(self, x, mask ): 
        att_ouput = self.multi_att(x, mask )
        x = self.norm1(x + self.dropout(att_ouput))
        ff = self.feedforward(x)
        return self.norm2(x + self.dropout(ff))
    



class TransformerEncoder(nn.Module): 

    def __init__(self): 
        pass 


    def forward(self): 
        pass 