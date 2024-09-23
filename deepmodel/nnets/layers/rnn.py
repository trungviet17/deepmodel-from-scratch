from layer import Linear, Layer
from activations import Tanh
import numpy as np 


class RNN(Layer): 

    def __init__(self, input_size: int, hidden_units: int, output_size: int,  nonlinearity: str = None ):
        """
        - input_size: The size of the input at each time step.
        - hidden_units: Number of hidden units in the recurrent hidden state.
        - output_size: Size of the output at each time step.
        """

        self.input_layer = Linear(input_size, hidden_units)
        self.hidden_layer = Linear(hidden_units, hidden_units)
        self.activation = Tanh()
        self.output_layer = Linear(hidden_units, output_size)

        # Store the hidden states for use in backpropagation
        self.hidden_states = []

    def forward(self, input_seq: np.array): 
        """
        Hàm forward 
        :param input_seq: Hàm nhận vào chỗi input với shape yêu câuù là  (sequence_length, input_size).
        :return: trả về một chuỗi giá trị sau khi thực hiện forward.
        """
        sequence_length = input_seq.shape[0]
        self.hidden_states = []  # Lưu trữ các hidden state sau mỗi thực hiện 
        res = []

        # khởi tạo giá trị output của a0 ban đầu
        out_hiddenlayer = np.zeros_like(self.hidden_layer.bias)

        for t in range(sequence_length): 
            # tính output xt 
            weighted_input = self.input_layer.forward(input_seq[t])
            # cộng thêm output at-1
            out_hiddenlayer = weighted_input + out_hiddenlayer
            # đóng gói state thông qua activation func
            state = self.activation.forward(out_hiddenlayer)
            # Lưu trữ các giá trị state cho thuật toán backward (BTT)
            self.hidden_states.append(state)
            # tính output 
            output = self.output_layer.forward(state)
            res.append(output)

            # Update hidden state cho bước tiếp theo 
            out_hiddenlayer = self.hidden_layer.forward(state)

        return np.array(res)

    def backward(self, output_error: np.array, learning_rate: float): 
        """
        Truyền ngược với thuật toán BTT
        :param output_error: error cuả ouputlayer 
        :param learning_rate: tốc độ learning .
        """
        sequence_length = len(self.hidden_states)
        state_error = np.zeros_like(self.hidden_states[0])
        hidden_layer_error = np.zeros_like(self.hidden_states[0])

        for t in reversed(range(sequence_length)):
            # lan truyền ngược trên output 
            output_layer_error = self.output_layer.backward(output_error[t], learning_rate)

            # tính tổng error của output và hiddenlayer trước đó 
            total_error = output_layer_error + hidden_layer_error
            # đạo hàm ngược activation function 
            state_error = self.activation.backward(total_error)

            # Backward + cập nhật tham số cho input_layer 
            self.input_layer.backward(state_error, learning_rate)
            hidden_layer_error = self.hidden_layer.backward(state_error, learning_rate)

    def reset_state(self):
        """Optional method to reset the hidden states between batches."""
        self.hidden_states = []
