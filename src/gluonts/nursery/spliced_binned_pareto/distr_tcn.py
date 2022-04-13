# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

# Implementation taken and modified from
# https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries, which was created
# with the following license.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


# Implementation of causal CNNs partly taken and modified from
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py, originally created
# with the following license.

# MIT License

# Copyright (c) 2018 CMU Locus Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn

from .tcn import TCNBlock

from torch.distributions.normal import Normal


class DistributionalTCN(torch.nn.Module):
    """
    Distributional Temporal Convolutional Network: a TCN to learn a time-
    varying distribution.

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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        channels: int,
        layers: int,
        bias: bool = True,
        fwd_time: bool = True,
        output_distr=Normal(torch.tensor([0.0]), torch.tensor([1.0])),
    ):

        super().__init__()

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
                    final=False,
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
                final=True,
            )
        )

        self.network = torch.nn.Sequential(*net_layers)
        self.output_distr = output_distr

    def forward(self, x):

        net_out = self.network(x)
        net_out_final = net_out[..., -1].squeeze()
        self.output_distr(net_out_final)

        return self.output_distr
