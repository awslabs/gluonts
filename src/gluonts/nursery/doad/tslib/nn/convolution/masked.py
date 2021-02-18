from typing import Optional
import math

import torch as pt
from torch import Tensor, BoolTensor
from torch import nn
from torch.nn import init

class MaskedConv1d(nn.Module):
    def __init__(self, 
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros'):
        super(MaskedConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, False, padding_mode,
        )
        self.mask_conv = nn.Conv1d(
            1, 1, kernel_size, stride, 
            padding, dilation, groups, False, 'zeros',
        )
        init.constant_(self.mask_conv.weight, 1.0/kernel_size)
        if bias:
            self.bias = nn.Parameter(Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
        
    def forward(self, 
            tensor: Tensor,
            mask: Optional[BoolTensor] = None):
        if mask is not None:
            mask = mask.unsqueeze(dim=1)
            tensor = tensor.masked_fill(mask, 0.0)
        output = self.conv(tensor)
        if mask is not None:
            scale = self.mask_conv(1-mask.float())
            scale = scale.add(1e-6).reciprocal()
            output = output * scale
        output = output + self.bias.view(-1, 1)
        return output
        
