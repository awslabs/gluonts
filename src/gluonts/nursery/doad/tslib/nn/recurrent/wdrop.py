from typing import Optional, List, Tuple, Union
from functools import partial

import torch as pt
from torch import Tensor, BoolTensor
from torch import nn
from torch.nn import functional as F



class WeightDrop(object):
    def __init__(self, dropout: float):
        self.dropout = dropout

    @staticmethod
    def _dummy_flatten_parameters(module: nn.RNNBase):
        pass

    @staticmethod
    def apply(module: nn.RNNBase, dropout: float):
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, WeightDrop):
                raise RuntimeError("module has registerd weight drop")
        assert module.num_layers == 1, 'the given module must have only one layer'
        weight = getattr(module, 'weight_hh_l0')
        del module._parameters['weight_hh_l0'] 
        module.register_parameter('weight_hh_l0_raw', nn.Parameter(weight.data))
        module.flatten_parameters = WeightDrop._dummy_flatten_parameters
        return module
        

    def __call__(self, module: nn.RNNBase, input: Tuple):
        raw_weight = getattr(module, 'weight_hh_l0_raw')
        weight = F.dropout(raw_weight, p=self.dropout, training=module.training)
        setattr(module, 'weight_hh_l0', weight)
        


class WeightDropRNN(nn.Module):
    def __init__(self, 
        cell_type: str,
        d_input: int,
        d_hidden: int,
        bidirectional: bool = False,
        dropout: float = 0.0,
        forget_bias: Optional[float] = 1.0,
    ):
        super(WeightDropRNN, self).__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.forget_bias = forget_bias

        self.rnn = WeightDrop.apply(self._get_cell(cell_type, bidirectional), dropout)
        self._reset_parameters()

    def _get_cell(self, cell_type: str, bidirectional: bool) -> nn.RNNCellBase:
        if cell_type == 'tanh':
            return nn.RNN(
                self.d_input, self.d_hidden, 1, 
                batch_first=True,
                bidirectional=bidirectional, 
                onlinearity='tanh',
            )
        elif cell_type == 'relu':
            return nn.RNN(
                self.d_input, self.d_hidden, 1,
                batch_first=True, 
                bidirectional=bidirectional, 
                nonlinearity='relu',
            )
        elif cell_type == 'lstm':
            return nn.LSTM(
                self.d_input, self.d_hidden, 1, 
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif cell_type == 'gru':
            return nn.GRU(
                self.d_input, self.d_hidden, 1, 
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            raise NotImplementedError(f'Unsupported rnn type "{rnn_type}"')

    def _reset_parameters(self):
        if (self.forget_bias is not None) and (isinstance(self.rnn, (nn.LSTM, nn.GRU))):
            with pt.no_grad():
                self.rnn.bias_ih_l0[self.d_hidden:2*self.d_hidden].fill_(self.forget_bias)
                self.rnn.bias_ih_l0[self.d_hidden:2*self.d_hidden].fill_(self.forget_bias)

    def forward(self, 
        input: Tensor,
        mask: Optional[BoolTensor] = None,
        time_step: Optional[Tensor] = None,
        init_time: Optional[Tensor] = None,
        init_state: Optional[Union[Tuple[Tensor, Tensor], Tensor]] = None
    ):
        if mask is not None:
            input = input.masked_fill(mask, 0.0)
        return self.rnn(input, init_state)
