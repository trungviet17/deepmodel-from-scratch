import torch 
import torch.nn as nn 


class Position_wise_feed_forward(nn.Module): 

    def __init__(self, d_model, hidden_units): 
        super(Position_wise_feed_forward, self).__init__()

        self.model = nn.Sequential(
            [
                nn.Linear(d_model, hidden_units), 
                nn.ReLU(), 
                nn.Linear(hidden_units, d_model)
            ]

        )

    def forward(self, x): 
        return self.model(x)