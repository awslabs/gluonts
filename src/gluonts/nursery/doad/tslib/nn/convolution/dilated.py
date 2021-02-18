from typing import List, Tuple, Optional
from functools import partial

import torch as pt
from torch import Tensor, BoolTensor
from torch import nn
from torch.nn import functional as F

from ..activations import GatedLinearUnit
from .masked import MaskedConv1d


class CausalDilatedResidue(nn.Module):
    def __init__(self,
        d_hidden: int,
        d_skip: int,
        kernel_size: int,
        dilation: int,
        padding: bool = True,
        dropout: float = 0.0,
        backward: bool = False,
    ):
        super(CausalDilatedResidue, self).__init__()
        self.d_hidden = d_hidden
        self.d_skip = d_skip
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.dropout = dropout
        self.backward = backward

        self.residual = nn.Conv1d(d_hidden, d_hidden, 1)
        self.skip = nn.Conv1d(d_hidden, d_skip, 1)
        
        self.conv = MaskedConv1d(
            self.d_hidden, 
            self.d_hidden*2, 
            self.kernel_size, 
            dilation=self.dilation,
        )
        self.act = GatedLinearUnit(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
            x: Tensor,
            mask: Optional[BoolTensor] = None) -> Tuple[Tensor, Tensor]:
        if self.padding:
            pad = (self.kernel_size-1)*self.dilation
            pad = (0, pad) if self.backward else (pad, 0)
            tensor = F.pad(x, pad, mode='constant', value=0.0)
            if mask is not None:
                mask = F.pad(mask, pad, mode='constant', value=False)
        else:
            tensor = x
        u = self.dropout(self.act(self.conv(tensor, mask)))
        skip = self.skip(u)
        residue = self.residual(u)
        if not self.padding:
            x = x[..., -residue.size(2):]
        output = x + residue
        return output, skip

    @property
    def receptive_field(self) -> int:
        return (self.kernel_size-1) * self.dilation + 1



class TempConvNetwork(nn.Module):
    def __init__(self,
        d_hidden: int,
        d_skip: int,
        kernel_sizes: List[int],
        dilations: List[int],
        padding: bool = True,
        bidirectional: bool = False,
        dropout: float = 0.0
    ):
        super(TempConvNetwork, self).__init__()
        assert len(kernel_sizes) == len(dilations)
        if bidirectional:
            assert d_hidden % 2 == 0 and d_skip % 2 == 0
        self.d_hidden = d_hidden
        self.d_skip = d_skip
        self.padding = padding
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.convs = nn.ModuleList()
        for ks, dil in zip(kernel_sizes, dilations):
            layer = nn.ModuleDict()
            if self.bidirectional:
                layer['fwd'] = CausalDilatedResidue(
                    self.d_hidden, 
                    self.d_skip,
                    ks,
                    dil,
                    self.padding,
                    dropout,
                    backward=False,
                )
                layer['bwd'] = CausalDilatedResidue(
                    self.d_hidden, 
                    self.d_skip,
                    ks,
                    dil,
                    self.padding,
                    dropout,
                    backward=True,
                )
            else:
                layer['fwd'] = CausalDilatedResidue(
                    self.d_hidden, 
                    self.d_skip,
                    ks,
                    dil,
                    self.padding,
                    dropout,
                    backward=False,
                )
            self.convs.append(layer)
    
            
    @property
    def n_layers(self) -> int:
        return len(self.convs)

    @property
    def receptive_fields(self) -> int:
        return sum([l['fwd'].receptive_field - 1 for l in self.convs]) + 1

    def forward(self, 
            x: Tensor,
            mask: Optional[BoolTensor] = None) -> Tensor:
        skip_outs = []
        output = x.transpose(1,2)
        for l, layer in enumerate(self.convs):
            output_f, skip_f = layer['fwd'](output, mask)
            if 'bwd' in layer:
                output_b, skip_b = layer['bwd'](output, mask)
                output = output_f + output_b
                skip = skip_f + skip_b
            else:
                output = output_f
                skip = skip_f
            skip_outs.append(skip)
            mask = None
        if not self.padding:
            skip_outs = [skip[..., -output.size(2):] for skip in skip_outs]
        skip_outs = pt.stack(skip_outs, dim=1)
        output = output.transpose(1,2)
        skip_outs = skip_outs.transpose(2,3)
        return output, skip_outs