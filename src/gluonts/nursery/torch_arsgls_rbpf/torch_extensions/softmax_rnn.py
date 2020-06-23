from typing import Tuple
import torch
from torch import nn


class SoftmaxRNN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, rnn_cls=nn.LSTM):
        super().__init__()
        self.rnn = rnn_cls(input_size=dim_in, hidden_size=dim_hidden)
        self.linear = nn.Linear(in_features=dim_hidden, out_features=dim_out)
        self.softmax = nn.Softmax()

    def forward(self, input: torch.Tensor, initial_state: (
            Tuple[torch.Tensor, torch.Tensor], torch.Tensor, None) = None):
        h, last_state = self.rnn(input=input, hx=initial_state)
        logits = self.linear(h)
        weights = self.softmax(logits)
        return weights, last_state
