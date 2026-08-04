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

# If you use this code in your work please cite:
# Multivariate Time Series Forecasting with Latent Graph Inference
# (https://arxiv.org/abs/2203.03423)

import torch
from torch import nn
import math


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nf: int = 64,
        n_layers: int = 2,
        act_last_layer: bool = False,
        act_fn=nn.SiLU(),
        residual=False,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.add_module(
            "layer_%d" % 0,
            torch.nn.Linear(in_features=in_channels, out_features=nf),
        )
        for i in range(1, self.n_layers - 1):
            self.add_module(
                "layer_%d" % i,
                torch.nn.Linear(in_features=nf, out_features=nf),
            )
        self.add_module(
            "layer_%d" % (self.n_layers - 1),
            torch.nn.Linear(in_features=nf, out_features=out_channels),
        )
        self.act_fn = act_fn
        self.residual = residual
        self.act_last_layer = act_last_layer * (1 - self.residual)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        for i in range(0, self.n_layers):
            x = self._modules["layer_%d" % i](x)
            if i < self.n_layers - 1 or self.act_last_layer:
                x = self.act_fn(x)
        if self.residual:
            x += input
        return x


class Conv1dResidual(nn.Module):
    def __init__(
        self, in_channels: int, nf: int = 64, actfn=nn.SiLU()
    ) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=nf, kernel_size=1
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=nf, out_channels=in_channels, kernel_size=1
        )
        self.actfn = actfn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        x = self.actfn(self.conv1(x))
        x = self.conv2(x)
        return inputs + x


class CNNResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        context_length: int,
        nf: int = 64,
        max_nf: int = 256,
        stride: int = 5,
    ) -> None:
        super().__init__()
        self.n_layers = math.ceil(math.log(context_length, stride))
        print("N layers")
        padding = int((-context_length) % stride)
        self.add_module("pad_%d" % 0, torch.nn.ConstantPad1d((padding, 0), 0))
        self.add_module(
            "cnn1d_%d" % 0,
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=nf,
                kernel_size=stride,
                stride=stride,
            ),
        )
        self.add_module(
            "cnn1d_residual_%d" % 0,
            Conv1dResidual(in_channels=nf, nf=int(nf / 2)),
        )
        current_length = int((context_length + padding) / stride)
        for i in range(1, self.n_layers):
            nf_prev = nf
            nf = min(nf * 2, max_nf)
            padding = int((-current_length) % stride)
            self.add_module(
                "pad_%d" % i, torch.nn.ConstantPad1d((padding, 0), 0)
            )
            self.add_module(
                "cnn1d_%d" % i,
                torch.nn.Conv1d(
                    in_channels=nf_prev,
                    out_channels=nf,
                    kernel_size=stride,
                    stride=stride,
                ),
            )
            self.add_module(
                "cnn1d_residual_%d" % i,
                Conv1dResidual(in_channels=nf, nf=int(nf / 2)),
            )
            current_length = int((current_length + padding) / stride)

        if nf * current_length != out_channels:
            self.lastlayer = torch.nn.Linear(nf * current_length, out_channels)
        else:
            self.lastlayer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.size() --> (bs, prediction_length)
        bs = x.size(0)
        for i in range(0, self.n_layers):
            x = self._modules["pad_%d" % i](x)
            x = self._modules["cnn1d_%d" % i](x)
            x = self._modules["cnn1d_residual_%d" % i](x)
        x = x.view(bs, -1)
        if self.lastlayer is not None:
            x = self.lastlayer(x)
        return x


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_length: int,
        pred_length: int,
    ) -> None:
        super(Linear, self).__init__()
        self.out_channels = out_channels
        self.pred_length = pred_length
        self.linear = nn.Linear(
            in_channels * input_length, out_channels * pred_length
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bs, n_nodes, seq_len, input_dim = inputs.size()
        inputs = inputs.reshape(bs, n_nodes, seq_len * input_dim)
        output = self.linear(inputs).view(
            bs, n_nodes, self.pred_length, self.out_channels
        )
        return output
