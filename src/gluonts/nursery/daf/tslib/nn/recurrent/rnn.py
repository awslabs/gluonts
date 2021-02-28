from typing import Optional, Union, Tuple, List

import torch as pt
from torch import Tensor, LongTensor, BoolTensor
from torch import nn

from .decay import ExpDecayRNN, RITS
from .wdrop import WeightDropRNN


class RNNDropout(nn.Dropout):
    def __init__(self, 
        p: float,
        batch_first: bool = True,
    ):
        super(RNNDropout, self).__init__(p, inplace=False)
        self.batch_first = batch_first

    def forward(self, x: Tensor):
        input_size = list(x.size())
        input_size[int(self.batch_first)] = 1
        mask = super(RNNDropout, self).forward(x.new_ones(*input_size))
        return x * mask



class StackedRecurrentNetwork(nn.Module):
    '''
    Multi-layer recurrent neural net.
    
    Args
    ----------
    rnns : List[ExpDecayRNN]
        A sequence of ExpDecayRNN modules to be stacked
    dropout : float
        dropout rate, by default 0.0
    bidirectional : bool
        add a backward RNN for each layer to make a BiRNN module, by default False
    '''
    def __init__(self, 
            d_input: int,
            d_hidden: int,
            n_layers: int,
            cell_type: str = 'gru',
            rnn_type: str = 'exp_decay',
            dropout: float = 0.0,
            bidirectional: bool = False,
            forget_bias: Optional[float] = 1.0,
            time_unit: float = 1.0) -> None:
        super(StackedRecurrentNetwork, self).__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.dropout = RNNDropout(dropout)
        self.bidirectional = bidirectional
        self.forget_bias = forget_bias
        self.time_unit = time_unit

        self.rnn = nn.ModuleList()
        d_input = self.d_input
        kwargs = {
            'cell_type': cell_type,
            'd_hidden': self.d_hidden,
            'bidirectional': self.bidirectional,
            'forget_bias': self.forget_bias,
        }
        for layer in range(self.n_layers):
            if rnn_type == 'exp_decay':
                rnn = ExpDecayRNN(d_input=d_input, time_unit=self.time_unit, **kwargs)
            elif rnn_type == 'rits':
                rnn = RITS(d_input=d_input, time_unit=self.time_unit, **kwargs)
            elif rnn_type == 'weight_drop':
                rnn = WeightDropRNN(d_input=d_input, dropout=dropout, **kwargs)
            else:
                raise NotImplementedError(f'Unsupported rnn type {rnn_type}')
            self.rnn.append(rnn)
            d_input = self.d_hidden*2 if self.bidirectional else self.d_hidden


    def forward(self, 
            input: Tensor,
            mask: Optional[BoolTensor] = None,
            time_step: Optional[Tensor] = None,
            init_time: Optional[Tensor] = None,
            init_state: Optional[Union[Tuple[Tensor, Tensor], Tensor]] = None):
        batch_size = input.size(0)
        if init_time is None:
            init_time = [None] * self.n_layers
        else:
            init_time = init_time.view(self.n_layers, -1, batch_size)
            init_time = init_time.unbind(dim=0)
        if init_state is None:
            init_state = [None] * self.n_layers
        elif isinstance(init_state, tuple):
            init_state = tuple(s.view(self.n_layers, -1, batch_size, self.d_hidden) for s in init_state)
            init_state = tuple(zip(*[s.unbind(dim=0) for s in init_state]))
        else:
            init_state = init_state.view(self.n_layers, -1, batch_size, self.d_hidden)
            init_state = init_state.unbind(dim=0)

        state_layers = []
        for layer in range(self.n_layers):
            rnn = self.rnn[layer]
            input, state = rnn(
                input, 
                mask=mask, 
                time_step=time_step, 
                init_time=init_time[layer], 
                init_state=init_state[layer],
            )
            if layer < self.n_layers - 1:
                input = self.dropout(input)
            mask = None
            state_layers.append(state)
        if isinstance(state_layers[0], tuple):
            state = tuple(pt.stack(ss, dim=0) for ss in zip(*state_layers))
            state = tuple(s.view(-1, *s.shape[-2:]) for s in state)
        else:
            state = pt.stack(state_layers, dim=0)
            state = state.view(-1, *state.shape[-2:])
        return input, state