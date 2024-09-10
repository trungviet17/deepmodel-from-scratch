import numpy as np 
import layer 
import loss
import activations
from typing import List


class Network: 
    """
    Class nay tong hop cac thanh phan duoc tao tu truoc do de tao thanh mang neural hoan chinh 
    """

    def __init__(self ):

        # danh sach layer cua lop 
        self.layers = [] 
        # ham loss 
        self.loss = None 
    

    # add layer into network 
    def add(self, layer: layer.Layer): 
        self.layers.append(layer)


    # 

