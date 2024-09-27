import torch.nn as nn 
import math 

from layer.attention import MultiHeadAttention 
from embedding.positional_emdbedding import PositionalEncoding
from layer.pointwise_feed_forward import Position_wise_feed_forward


class DecoderLayer(nn.Module): 

    def __init__(self, d_model, num_head, d_ff, dropout): 
        super(DecoderLayer, self).__init__()
        self.masked_multi_att = MultiHeadAttention(d_model, num_head)
        self.multi_att = MultiHeadAttention(d_model, num_head)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.feedforward = Position_wise_feed_forward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)



    def forward(self, x, mask):

        att_output = self.masked_multi_att(x, mask)
        x = self.norm1(x + self.dropout(att_output))

        att_output = self.multi_att(x, mask)
        x = self.norm2(x + self.dropout(att_output))

        ff_out = self.feedforward(x)
        x = self.norm3(x + self.dropout(att_output))

        return x 
    

class TransformerDecoder(nn.Module): 

    def __init__(self, d_model, num_head, d_ff, dropout, vocab_size, max_seq_length, num_layers):
        super(TransformerDecoder, self).__init__()
        self.output_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = PositionalEncoding(d_model, max_seq_length)
        self.decoders = nn.ModuleList([DecoderLayer(d_model, num_head, d_ff, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, vocab_size)


    def forward(self, x, mask): 
        

        x = self.output_embedding(x)
        x = self.positional_embedding(x)

        for layer in self.decoders:
            x = layer(x, mask)

        return x 

