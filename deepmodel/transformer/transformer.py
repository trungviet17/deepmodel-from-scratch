import torch 
import torch.nn as nn 
from blocks.decoder import TransformerDecoder
from blocks.encoder import TransformerEncoder


class Transformer(nn.Module): 

    def __init__(self, d_model, num_head, d_ff, dropout, vocab_size, max_seq_length, num_layers): 
        
        self.encoder_block = TransformerEncoder(vocab_size, d_model, max_seq_length, 
                                                num_layers, d_ff, dropout, num_head)
        self.decoder_block = TransformerDecoder(d_model, num_head, d_ff, dropout, 
                                                vocab_size, max_seq_length, num_layers)
        


    def generate_mask(self, src : torch.Tensor, tar: torch.Tensor): 
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tar_mask = (tar != 0).unsqueeze(1).unsqueeze(2)
        seq_length = tar.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tar_mask = tar_mask & nopeak_mask
        return src_mask, tar_mask 



    def forward(self, src, tar): 
        
        src_mask , tar_mask = self.generate_mask(src, tar)

        src_output = self.encoder_block(src, src_mask)

        tar_output = self.decoder_block(tar, src_mask, tar_mask)

        