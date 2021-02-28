from typing import Optional, Union, Tuple, List

import torch as pt
from torch import nn
from torch.nn import init, functional as F

from tslib.nn.diffeq import odeint


class GradNet(nn.Module):
    def __init__(self,
            d_input: int,
            d_output: int,
            d_hidden: int,
            n_layers: int):
        super(GradNet).__init__()
        network = [nn.Linear(d_input+1, d_hidden)]
        for l in range(n_layers):
            network.append(nn.Tanh())
            network.appedd(nn.Linear(d_hidden, d_hidden))
        network.append(nn.Linear(d_hidden, d_output))
        self.network = nn.Sequential(*network)

    def forward(self, time: Tensor, value: Tensor):
        x = pt.cat([value, time], dim=-1)
        return self.network(x)


class ODERNN(nn.Module):
    def __init__(self,
            grad_net: nn.Module,
            cell_type: str,
            d_input: int,
            d_hidden: int,
            bidirectional: bool = False,
            forget_bias: Optional[float] = 1.0,
            time_unit: float = 1.0,
            rtol: float = 1e-4,
            atol: float = 1e-5):
        super(ODERNN, self).__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.bidirectional = bidirectional
        self.forget_bias = forget_bias
        self.time_unit = time_unit

        self.grad_net = GradNet(self.d_hidden, self.d_hidden, d_grad, n_grad_layers)
        self.cell = self._get_cell(cell_type)
        self.bw_cell = self._get_cell(cell_type) if self.bidirectional else None
        self._reset_parameters()

    def _get_cell(self, cell_type: str) -> nn.RNNCellBase:
        if cell_type == 'tanh':
            return nn.RNNCell(self.d_input, self.d_hidden, nonlinearity='tanh')
        elif cell_type == 'relu':
            return nn.RNNCell(self.d_input, self.d_hidden, nonlinearity='relu')
        elif cell_type == 'lstm':
            return nn.LSTMCell(self.d_input, self.d_hidden)
        elif cell_type == 'gru':
            return nn.GRUCell(self.d_input, self.d_hidden)
        else:
            raise NotImplementedError(f'Unsupported rnn type "{rnn_type}"')

    def _reset_parameters(self):
        if not (self.forget_bias is None or isinstance(self.cell, nn.RNNCell)):
            with pt.no_grad():
                self.cell.bias_ih[self.d_hidden:2*self.d_hidden].fill_(self.forget_bias)
                self.cell.bias_ih[self.d_hidden:2*self.d_hidden].fill_(self.forget_bias)

    def forward(self, 
            input: Tensor,
            mask: Optional[BoolTensor],
            time_step: Optional[Tensor],
            init_time: Optional[Tensor] = None,
            init_state: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None):
        
        