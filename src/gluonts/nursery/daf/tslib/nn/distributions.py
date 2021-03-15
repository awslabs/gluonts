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


from typing import List, Tuple

import torch as pt
from torch import Tensor
from torch import nn
from torch.distributions import Distribution

from .activations import PositiveSoftplus


def distribution_cat(distributions: List[Distribution], dim=0) -> Distribution:
    args = list(distributions[0].arg_constraints.keys())
    if "probs" in args and "logits" in args:
        try:
            _ = getattr(distributions[0], "probs")
            args.remove("logits")
        except AttributeError:
            args.remove("probs")
    concat_args = {
        arg: pt.cat([getattr(d, arg) for d in distributions], dim=dim)
        for arg in args
    }
    return type(distributions[0])(**concat_args)


class GaussianLayer(nn.Module):
    def __init__(self, d_hidden: int, d_data: int):
        super().__init__()
        self.mean = nn.Linear(d_hidden, d_data)
        self.var = nn.Sequential(
            nn.Linear(d_hidden, d_data), PositiveSoftplus(1e-3)
        )

    def forward(self, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        mu = self.mean(hidden)
        sigma = self.var(hidden)
        return mu, sigma
