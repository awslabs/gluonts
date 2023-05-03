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


import torch as pt
from torch import Tensor, nn
from torch.nn import init, functional as F


class SimpleDiscriminator(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        n_hidden: int = 1,
    ) -> None:
        super(SimpleDiscriminator, self).__init__()
        self.d_model = d_model
        self.d_hidden = d_model * n_hidden
        self.n_layer = n_layer

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for layer in range(n_layer):
            d_input = self.d_model if layer == 0 else self.d_hidden
            d_output = self.d_model if layer == n_layer - 1 else self.d_hidden
            self.weights.append(nn.Parameter(Tensor(d_output, d_input)))
            self.biases.append(nn.Parameter(Tensor(d_output)))
        self.class_weight = nn.Parameter(Tensor(1, self.d_model))
        self.class_bias = nn.Parameter(Tensor(1))

        self._reset_parameters()

    def _reset_parameters(self):
        for layer in range(self.n_layer):
            init.normal_(self.weights[layer], std=1e-1)
            init.zeros_(self.biases[layer])
        init.normal_(self.class_weight, std=1e-1)
        init.zeros_(self.class_bias)

    def forward(self, x: Tensor):
        for layer in range(self.n_layer):
            x = F.linear(x, self.weights[layer], self.biases[layer])
            x = F.gelu(x)
        x = F.linear(x, self.class_weight, self.class_bias)
        x = pt.sigmoid(x)
        x = x.add(1e-10).log()
        return x
