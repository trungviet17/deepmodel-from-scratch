from layer import Linear, Layer
from activations import Tanh
import numpy as np 


class RNN(Layer): 


    def __init__(self, input_size: int, hidden_units: int, output_size: int,  nonlinearity: str = None ):
        """
        - hàm nhận vào input shape : chiều dài của toàn bộ chuỗi 
        - hàm nhận hidden units : số phần tử của hidden unit h trong mạng 
        """

        self.input_layer = Linear(input_size, hidden_units)
        self.hidden_layer = Linear(hidden_units, hidden_units)
        self.activation = Tanh()
        self.output_layer = Linear(hidden_units, output_size)


    def forward(self, input: np.array): 
        """
        Hàm forward 
        """


        res = []
        # khởi tạo output của a0 = 0 
        out_hiddenlayer = np.zeros_like(self.hidden_layer.bias)

        for i in range(input.shape[0]): 
            # tính output của xi 
            weighted_input = self.input_layer.forward(input[i])
            # cộng tổng output của xi với output ai-1 trước đó 
            out_hiddenlayer = weighted_input + out_hiddenlayer
            # tính state thông qua lấy activation function 
            state = self.activation.forward(out_hiddenlayer)
            # truyền state qua output layer để lấy kết quả 
            output = self.output_layer.forward(state)

            # tính toán output của ai 
            out_hiddenlayer = self.hidden_layer.forward(state)

            res.append(output)

        return output


    def backward(self, output_error: np.array, learning_rate : float): 
        """
        Thực hiện tính backward với input đầu vào là loss của output và learning rate
        """
        # tính loss khi đi qua outputlayer 

        state_error = self.output_layer.backward(output_error= output_error, learning_rate= learning_rate)





        pass 
