# QUESTION: which license goes here? "Amazon" one (as in spliced_binned_pareto.py), or "Implementation taken and modified from" (as in tcn.py)

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn



from tcn import TCNBlock

from torch.distributions.normal import Normal


class DistributionalTCN(torch.nn.Module):
    """ Distributional Temporal Convolutional Network: a TCN to learn a time-varying distribution.

    Composed of a sequence of causal convolution blocks.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    Args:
        in_channels : Number of input channels, typically the dimensionality of the time series 
        out_channels : Number of output channels, typically the number of parameters in the time series distribution
        kernel_size : Kernel size of the applied non-residual convolutions.
        channels : Number of channels processed in the network and of output channels, 
                typically equal to out_channels for simplicity, expand for better performance.
        layers : Depth of the network.
        bias : If True, adds a learnable bias to the convolutions.
        fwd_time : If True the network is the relation relation if from past to future (forward),
                if False, the relation from future to past (backward).
        output_distr: Distribution whose parameters will be specified by the network output
    """
    def __init__(self,
        in_channels:int,
        out_channels:int,
        kernel_size:int,
        channels:int,
        layers:int,
        bias:bool=True,
        fwd_time:bool=True,
        output_distr=Normal(torch.tensor([0.]), torch.tensor([1.])),
        ): 

        super(DistributionalTCN, self).__init__()

        self.out_channels = out_channels
        
        # Temporal Convolution Network
        layers = int(layers)
        
        net_layers = []  # List of sequential TCN blocks
        dilation_size = 1  # Initial dilation size

        for i in range(layers):
            in_channels_block = in_channels if i == 0 else channels
            net_layers.append(
                TCNBlock(
                in_channels=in_channels_block,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=dilation_size,
                bias=bias,
                fwd_time=fwd_time,
                final=False
                )
            )
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        net_layers.append(
            TCNBlock(
                in_channels=channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                dilation=dilation_size,
                bias=bias,
                fwd_time=fwd_time,
                final=True
            )
        )

        self.network = torch.nn.Sequential( *net_layers )
        self.output_distr = output_distr
        
        
    def forward(self, x):
        
        net_out = self.network(x)
        net_out_final = net_out[..., -1].squeeze()
        self.output_distr(net_out_final)
    
        return self.output_distr
        
        