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


from typing import List, Optional
from collections import deque
import math

import torch as pt
from torch import Tensor
from torch import nn
from torch.distributions import Distribution
from torch.distributions import Normal as Gaussian


class ResidualBlock(nn.Module):
    """
    Network module wrapped by residual connection

    Args
    ----------
    residual_network: nn.Module
        the main module to be wrapped
    skip_network: nn.Module, optional
        the auxiliary module to guarantee valid addition
    """

    def __init__(
        self,
        residual_network: nn.Module,
        skip_network: Optional[nn.Module] = None,
    ):
        super(ResidualBlock, self).__init__()
        self.residual_network = residual_network
        if skip_network is None:
            self.skip_network = nn.Identity()
        else:
            self.skip_network = skip_network

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        skip = self.skip_network(x)
        x = self.residual_network(x, *args, **kwargs)
        return skip + x


class Concatenate(nn.Module):
    def __init__(self, dim: int):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, inputs: List[Tensor]) -> Tensor:
        return pt.cat(inputs, dim=self.dim)


class LockedDropout(nn.Module):
    def __init__(self, dropout=0, lock_dim=None, inplace=False):
        super(LockedDropout, self).__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=inplace)
        self.lock_dim = lock_dim

    def forward(self, x):
        if self.lock_dim is None:
            return self.dropout(x)
        else:
            mask = self.dropout(
                pt.ones_like(x).narrow(self.lock_dim, 0, 1)
            ).expand_as(x)
            return mask * x


class DilatedQueue(nn.Module):
    def __init__(self, maxlen):
        super(DilatedQueue, self).__init__()
        self.maxlen = maxlen
        self.register_buffer("data", None)

    def _reset(self):
        self.data = None

    def initialize(self, x):
        self._reset()
        if x.size(2) < self.maxlen:
            x = nn.functional.pad(x, (self.maxlen - x.size(2), 0))
        else:
            x = x.narrow(2, x.size(2) - self.maxlen, self.maxlen)
        self.data = x

    def append(self, x):
        data = pt.cat([self.data, x], dim=2)
        self.data = data.narrow(2, data.size(2) - self.maxlen, self.maxlen)
