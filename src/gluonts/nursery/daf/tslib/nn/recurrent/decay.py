from typing import Optional, Union, Tuple, List

import torch as pt
from torch import Tensor, LongTensor, BoolTensor
from torch import nn
from torch.nn import functional as F
from torch.nn import init



class ExpDecayRNN(nn.Module):
    '''
    Exponentially decay the RNN state according to the time gap between two observations.
    It wraps a nn.RNNCell instance and takes optional `time_step` and `mask` argument.
    
    Parameters
    ----------
    cell : torch.nn.RNNCellBase
        a common RNNCell module
    forget_bias : Optional[float], optional
        bias added to forget/reset gate, by default 1.0
    time_unit: float
        unit time interval, by default 1.0
    '''
    def __init__(self, 
            cell_type: str,
            d_input: int,
            d_hidden: int,
            bidirectional: bool = False,
            forget_bias: Optional[float] = 1.0,
            time_unit: float = 1.0) -> None:
        super(ExpDecayRNN, self).__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.bidirectional = bidirectional
        self.forget_bias = forget_bias
        self.time_unit = time_unit

        self.cell = self._get_cell(cell_type)
        self.bw_cell = self._get_cell(cell_type) if self.bidirectional else None
        self._init_parameters()
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

    def _init_parameters(self):
        self._input_decay_weight = nn.Parameter(Tensor(self.d_input, self.d_input))
        self._input_decay_bias = nn.Parameter(Tensor(self.d_input))
        self._hidden_decay_weight = nn.Parameter(Tensor(self.d_hidden, 1))
        self._hidden_decay_bias = nn.Parameter(Tensor(self.d_hidden))

    def _reset_parameters(self):
        init.ones_(self._hidden_decay_weight)
        init.ones_(self._input_decay_weight)
        init.zeros_(self._hidden_decay_bias)
        init.zeros_(self._input_decay_bias)
        if not (self.forget_bias is None or isinstance(self.cell, nn.RNNCell)):
            with pt.no_grad():
                self.cell.bias_ih[self.d_hidden:2*self.d_hidden].fill_(self.forget_bias)
                self.cell.bias_ih[self.d_hidden:2*self.d_hidden].fill_(self.forget_bias)

    def _init_hidden(self, tensor: Tensor):
        state = tensor.new_zeros((tensor.size(0), self.d_hidden))
        if isinstance(self.cell, nn.LSTMCell):
            return state, state.clone()
        else:
            return state

    def _compute_time_delta(self,
            time_step: Tensor,
            init_time: Optional[Tensor]):
        if init_time is None:
            init_time = time_step[:, 0]
        init_time = init_time.unsqueeze(dim=1)
        time_step = pt.cat([init_time, time_step], dim=1)
        time_delta = time_step[:, 1:] - time_step[:, :-1]
        return time_delta

    def _process_missing_values(self,
            input: Tensor,
            time_delta: Tensor,
            mask: BoolTensor):
        time_delta = time_delta.unsqueeze(dim=2).repeat(1,1,input.size(2))
        prev_input = pt.zeros_like(input[:,0])
        prev_time_delta = pt.zeros_like(time_delta[:,0])
        input_delta = []
        forward_filled_input = []
        for step in range(input.size(1)):
            current_mask = mask[:,step]
            current_input = input[:,step]
            current_time_delta = time_delta[:,step]
            prev_input = prev_input.where(current_mask, current_input)
            forward_filled_input.append(prev_input)
            prev_time_delta = prev_time_delta + current_time_delta
            input_delta.append(prev_time_delta)
            prev_time_delta = prev_time_delta.masked_fill(~current_mask, 0.0)
        input_delta = pt.stack(input_delta, dim=1)
        forward_filled_input = pt.stack(forward_filled_input, dim=1)
        input_mean = input.masked_fill(mask, 0.0).sum(dim=1).div(mask.to(pt.float).sum(dim=1).clamp_min(1.0)).unsqueeze(dim=1)
        input_delta = pt.clamp(self.time_unit * (input_delta - 1), 0.0, 1000.0)
        fill_weight = F.relu(F.linear(input_delta, self._input_decay_weight, self._input_decay_bias)).mul(-1.0).exp()
        filled_input = forward_filled_input * fill_weight + (1 - fill_weight) * input_mean
        filled_input = pt.where(mask, filled_input, input)
        return filled_input

    def _pass(self, 
            cell: nn.RNNBase,
            input: Tensor,
            mask: Optional[BoolTensor], 
            time_delta: Tensor,
            state: Union[Tuple[Tensor, Tensor], Tensor]):
        hidden = []
        step_size = input.size(1)
        for step in range(step_size):
            x = input[:, step]
            td = time_delta[:, step, None]
            decay_rate = F.relu(F.linear(td, self._hidden_decay_weight, self._hidden_decay_bias)).mul(-1.0).exp()
            if isinstance(state, tuple):
                state = tuple(s * decay_rate for s in state)
            else:
                state = state * decay_rate
            # if mask is not None:
            #     m = mask[:, step]
            #     x = pt.cat([x, m], dim=-1)
            # else:
            #     x = pt.cat([x, pt.zeros_like(x)], dim=-1)
            state = cell(x, state)
            if isinstance(state, tuple):
                hidden.append(state[0])
            else:
                hidden.append(state)
        hidden = pt.stack(hidden, dim=1)
        return hidden, state

    def fwpass(self,
            input: Tensor,
            mask: Optional[BoolTensor], 
            time_step: Optional[Tensor],
            init_time: Optional[Tensor],
            init_state: Optional[Union[Tuple[Tensor, Tensor], Tensor]]):
        if time_step is None:
            time_delta = pt.ones_like(input[...,0])
        else:
            time_delta = self._compute_time_delta(time_step, init_time)
        if mask is not None:
            input = self._process_missing_values(input, time_delta, mask)
            # mask = 1.0 - mask.to(pt.float)
        time_delta = pt.clamp(self.time_unit * time_delta, 0.0, 1000.0).log()
        if init_state is None:
            state = self._init_hidden(input)
        else:
            state = init_state
        return self._pass(self.cell, input, mask, time_delta, state)

    def bwpass(self, 
            input: Tensor,
            mask: Optional[BoolTensor], 
            time_step: Optional[Tensor],
            init_time: Optional[Tensor],
            init_state: Optional[Union[Tuple[Tensor, Tensor], Tensor]]):
        input = input.flip(1)
        if time_step is None:
            time_delta = pt.ones_like(input[...,0])
        else:
            time_step = time_step.flip(1)
            time_delta = self._compute_time_delta(time_step, init_time)
            time_delta = time_delta.mul(-1.0)
        if mask is not None:
            # current pytorch version does not support flip for BoolTensor
            # mask = mask.flip(1)
            mask = mask.to(pt.float).flip(1).to(pt.bool)
            input = self._process_missing_values(input, time_delta, mask) 
        if init_state is None:
            state = self._init_hidden(input)
        else:
            state = init_state
        hidden, state = self._pass(self.bw_cell, input, mask, time_delta, state)
        hidden = hidden.flip(1)
        return hidden, state
        
    def forward(self,
            input: Tensor,
            mask: Optional[BoolTensor] = None,
            time_step: Optional[Tensor] = None,
            init_time: Optional[Tensor] = None,
            init_state: Optional[Union[Tuple[Tensor, Tensor], Tensor]] = None):
        init_time_fw = None if init_time is None else init_time[0]
        if init_state is None:
            init_state_fw = None
        elif isinstance(init_state, tuple):
            init_state_fw = tuple(s[0] for s in init_state)
        else:
            init_state_fw = init_state[0]
        hidden_fw, state_fw = self.fwpass(input, mask, time_step, init_time_fw, init_state_fw)
        if self.bidirectional:
            init_time_bw = None if init_time is None else init_time[1]
            if init_state is None:
                init_state_bw = None
            elif isinstance(init_state, tuple):
                init_state_bw = tuple(s[1] for s in init_state)
            else:
                init_state_bw = init_state[1]
            hidden_bw, state_bw = self.bwpass(input, mask, time_step, init_time_bw, init_state_bw)
            hidden = pt.cat([hidden_fw, hidden_bw], dim=-1)
            if isinstance(state_fw, tuple):
                state = tuple(pt.stack(ss, dim=0) for ss in zip(state_fw, state_bw))
            else:
                state = pt.stack([state_fw, state_bw], dim=0)
        else:
            hidden = hidden_fw
            if isinstance(state_fw, tuple):
                state = tuple(pt.unsqueeze(s, dim=0) for s in state_fw)
            else:
                state = state_fw.unsqueeze(dim=0)
        return hidden, state
        


class RITS(ExpDecayRNN):
    '''
    Recurrent Imputation for Time Series, cf. http://papers.nips.cc/paper/7911-brits-bidirectional-recurrent-imputation-for-time-series
    A follower on ExpDecay by replacing empirical imputation with regressive estimates
    '''
    @staticmethod
    def diag_mask(module, inputs):
        mask = 1 - pt.eye(module.d_input, module.d_input, requires_grad=False).to(pt.float)
        setattr(module, '_feat_regress_masked_weight', module._feat_regress_weight * mask)

    def _init_parameters(self) -> None:
        self._input_decay_weight = nn.Parameter(Tensor(self.d_input, 1))
        self._input_decay_bias = nn.Parameter(Tensor(self.d_input))
        self._hidden_decay_weight = nn.Parameter(Tensor(self.d_hidden, 1))
        self._hidden_decay_bias = nn.Parameter(Tensor(self.d_hidden))
        self._hist_regress_weight = nn.Parameter(Tensor(self.d_input, self.d_hidden))
        self._hist_regress_bias = nn.Parameter(Tensor(self.d_input))
        self._feat_regress_weight = nn.Parameter(Tensor(self.d_input, self.d_input))
        self._feat_regress_bias = nn.Parameter(Tensor(self.d_input))
        self.register_forward_pre_hook(RITS.diag_mask)
        
    def _reset_parameters(self):
        super(RITS, self)._reset_parameters()
        a = self.d_input**-0.5
        init.uniform_(self._hist_regress_weight, -a, a)
        init.zeros_(self._hist_regress_bias)
        init.uniform_(self._feat_regress_weight, -a, a)
        init.zeros_(self._feat_regress_bias)

    def _process_missing_values(self, input: Tensor, *args):
        return input

    def _pass(self, 
            cell: nn.RNNBase,
            input: Tensor,
            mask: Optional[BoolTensor], 
            time_delta: Tensor,
            state: Union[Tuple[Tensor, Tensor], Tensor]):
        hidden = []
        step_size = input.size(1)
        for step in range(step_size):
            x = input[:, step]
            td = time_delta[:, step]
            decay_hidden = F.relu(F.linear(td, self._hidden_decay_weight, self._hidden_decay_bias)).mul(-1.0).exp()
            if mask is not None:
                m = mask[:, step]
                decay_input = F.relu(F.linear(td, self._input_decay_weight, self._input_decay_bias)).mul(-1.0).exp()
                x_hist = F.linear(state, self._hist_regress_weight, self._hist_regress_bias)
                x_feat = F.linear(pt.where(m, x_hist, x), self._feat_regress_masked_weight, self._feat_regress_bias)
                x_hat = decay_hidden * x_hist + (1-decay_hidden) * x_feat
                x = pt.where(m, x_hat, x)
            #     m = 1.0 - m.to(pt.float)
            #     x = pt.cat([x, m], dim=-1)
            # else:
            #     x = pt.cat([x, pt.zeros_like(x)], dim=-1)
            if isinstance(state, tuple):
                state = tuple(s * decay_hidden for s in state)
            else:
                state = state * decay_hidden
            state = self.cell(x, state)
            if isinstance(state, tuple):
                hidden.append(state[0])
            else:
                hidden.append(state)
        hidden = pt.stack(hidden, dim=1)
        return hidden, state