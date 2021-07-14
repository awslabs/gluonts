# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Original implementation taken and modified from
# https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
# distributed under the Apache Licence 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn.functional as F


class Chomp1d(torch.nn.Module):
    """Removes leading or trailing elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.

    Args:
        chomp_size : Number of elements to remove.
    """
    def __init__(self, chomp_size:int, last:bool=True):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TCNBlock(torch.nn.Module):
    """ Temporal Convolutional Network block.

    Composed sequentially of two causal convolutions (with leaky ReLU activation functions),
    and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    Args:
        in_channels : Number of input channels.
        out_channels : Number of output channels.
        kernel_size : Kernel size of the applied non-residual convolutions.
        dilation : Dilation parameter of non-residual convolutions.
        final : If True, the last activation function is disabled.
    """
    def __init__(
            self,
            in_channels:int,
            out_channels:int,
            kernel_size:int,
            dilation:int,
            final:bool = False,
        ):

        super(TCNBlock, self).__init__()

        in_channels = int(in_channels)
        kernel_size=int(kernel_size)
        out_channels=int(out_channels)
        dilation=int(dilation)

        # Computes left padding so that the applied convolutions are causal
        padding = int( (kernel_size - 1) * dilation )

        # First causal convolution
        conv1_pre = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation,
        )
        conv1 = torch.nn.utils.weight_norm( conv1_pre )
        
        # The truncation makes the convolution causal
        chomp1 = Chomp1d( chomp_size=padding )

        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2_pre = torch.nn.Conv1d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation,
        )
        conv2 = torch.nn.utils.weight_norm( conv2_pre )
        chomp2 = Chomp1d( chomp_size=padding )
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        ) if in_channels != out_channels else None

        # Final activation function
        self.activation = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.activation is None:
            return out_causal + res
        else:
            return self.activation(out_causal + res)


class TCN(torch.nn.Module):
    """ Temporal Convolutional Network.

    Composed of a sequence of causal convolution blocks.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    Args:
        in_channels : Number of input channels.
        out_channels : Number of output channels.
        kernel_size : Kernel size of the applied non-residual convolutions.
        channels : Number of channels processed in the network and of output
            channels.
        layers : Depth of the network.
    """
    def __init__(
            self,
            in_channels:int,
            out_channels:int,
            kernel_size:int,
            channels:int,
            layers:int,
        ):

        super(TCN, self).__init__()

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
                    final=False
                )
            )
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        net_layers.append(
            TCNBlock(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation_size,
                final=True
            )
        )

        self.network = torch.nn.Sequential( *net_layers )

    def forward(self, x):
        return self.network(x)


class TCNEncoder(torch.nn.Module):
    """ Encoder of a time series using a Temporal Convolution Network (TCN).
    
    The computed representation is the output of a fully connected layer applied
    to the output of an adaptive max pooling layer applied on top of the TCN,
    which reduces the length of the time series to a fixed size.

    Takes as input a three-dimensional tensor (`B`, `C_in`, `L`) where `B` is the
    batch size, `C_in` is the number of input channels, and `L` is the length of
    the input. Outputs a two-dimensional tensor (`B`, `C_out`), `C_in` is the
    number of input channels C_in=tcn_channels*

    Args:
        in_channels : Number of input channels.
        out_channels : Dimension of the output representation vector.
        kernel_size : Kernel size of the applied non-residual convolutions.
        tcn_channels : Number of channels manipulated in the causal CNN.
        tcn_layers : Depth of the causal CNN.
        tcn_out_channels : Number of channels produced by the TCN.
            The TCN outputs a tensor of shape (B, tcn_out_channels, T)
        maxpool_out_channels : Fixed length to which each channel of the TCN
            is reduced.
        normalize_embedding : Normalize size of the embeddings
    """
    def __init__(self,
        in_channels:int,
        out_channels:int,
        kernel_size:int,
        tcn_channels:int,
        tcn_layers:int,
        tcn_out_channels:int,
        maxpool_out_channels:int=1,
        normalize_embedding:bool=True,
    ):

        super(TCNEncoder, self).__init__()
        tcn = TCN(
            in_channels=in_channels,
            out_channels=tcn_out_channels,
            kernel_size=kernel_size,
            channels=tcn_channels,
            layers=tcn_layers,
        )

        maxpool_out_channels = int(maxpool_out_channels)
        maxpooltime = torch.nn.AdaptiveMaxPool1d( maxpool_out_channels )
        flatten = torch.nn.Flatten() # Flatten two and third dimensions (tcn_out_channels and time)
        fc = torch.nn.Linear( tcn_out_channels * maxpool_out_channels , out_channels )
        self.network = torch.nn.Sequential(
            tcn, maxpooltime, flatten, fc
        )
            
        self.normalize_embedding = normalize_embedding

    def forward(self, x):
        u = self.network(x)
        if self.normalize_embedding:
            return F.normalize(u, p=2, dim=1)
        else:
            return u
