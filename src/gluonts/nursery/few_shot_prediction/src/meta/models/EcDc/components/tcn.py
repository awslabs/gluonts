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
#
# Code from:
# https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/blob/master/networks/causal_cnn.py
# which is based on:
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py, originally created
# with the following license.
#
# MIT License
# Copyright (c) 2018 CMU Locus Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn.functional as F


class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """

    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(2)


class CausalConvolution(torch.nn.Module):
    """
    A single causal convolution applies the causal convolution itself, weight normalization, and
    an activation function.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        """
        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            kernel_size: The kernel size to use for the convolution.
            dilation: The dilation parameter to use for the causal convolution.
        """
        super().__init__()

        conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.padding = (kernel_size - 1) * dilation
        self.normalized_conv = torch.nn.utils.weight_norm(conv)
        self.activation = torch.nn.LeakyReLU()

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Computes the causal convolution for the provided sequence. Inputs are padded such that
        the output sequence length

        Args:
            sequences: Tensor of shape `[batch_size, in_channels, sequence_length]`.

        Returns:
            Tensor of shape `[batch_size, out_channels, sequence_length]`.
        """
        # Pad sequences on the left only
        padded_sequences = F.pad(sequences, pad=(self.padding, 0))
        # After running the convolution, the length of the sequence equals the input's
        out = self.normalized_conv(padded_sequences)
        return self.activation(out)


class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels (= number of dimensions
    of the multivariate time series), and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).
    Parameters
    ----------
    in_channels : Number of input channels.
    out_channels : Number of output channels.
    kernel_size : Kernel size of the applied non-residual convolutions.
    dilation : Dilation parameter of non-residual convolutions.
    final : Disables, if True, the last activation function.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        final: bool = False,
    ):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = CausalConvolution(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )

        # Second causal convolution
        conv2 = CausalConvolution(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )

        # Causal network
        self.causal = torch.nn.Sequential(conv1, conv2)

        # Residual connection
        self.upordownsample = (
            torch.nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_causal = self.causal(x)
        if self.upordownsample is None:
            res = x
        else:
            res = self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).
    Parameters
    ----------
    in_channels : Number of input channels.
    channels : Number of channels processed inside the network and of output
           channels.
    depth : Depth of the network.
    out_channels : Number of output channels.
    kernel_size : Kernel size of the applied non-residual convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        depth: int,
        out_channels: int,
        kernel_size: int,
    ):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [
                CausalConvolutionBlock(
                    in_channels_block, channels, kernel_size, dilation_size
                )
            ]
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [
            CausalConvolutionBlock(
                channels, out_channels, kernel_size, dilation_size
            )
        ]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).
    Parameters
    ----------
    in_channels : Number of input channels.
    channels : Number of channels manipulated in the causal CNN.
    depth : Depth of the causal CNN.
    reduced_size : Fixed length to which the output time series of the
           causal CNN is reduced.
    out_channels : Number of output channels.
    kernel_size : Kernel size of the applied non-residual convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        depth: int,
        out_channels: int,
        kernel_size: int,
    ):
        super(CausalCNNEncoder, self).__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, depth, out_channels, kernel_size
        )
        self.network = torch.nn.Sequential(
            causal_cnn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.network(x)
        return u
