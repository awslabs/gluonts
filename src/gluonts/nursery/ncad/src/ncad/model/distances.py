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


import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class CosineDistance(nn.Module):
    r"""Returns the cosine distance between :math:`x_1` and :math:`x_2`, computed along dim."""

    def __init__(
        self,
        dim: int = 1,
        keepdim: bool = True,
    ) -> None:

        super().__init__()
        self.dim = int(dim)
        self.keepdim = bool(keepdim)
        self.eps = 1e-10

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        # Cosine of angle between x1 and x2
        cos_sim = F.cosine_similarity(x1, x2, self.dim, self.eps)
        dist = -torch.log((1 + cos_sim) / 2)

        if self.keepdim:
            dist = dist.unsqueeze(dim=self.dim)
        return dist


class LpDistance(nn.Module):
    r"""Returns the Lp norm between :math:`x_1` and :math:`x_2`, computed along dim."""

    def __init__(
        self,
        p: int = 2,
        dim: int = 1,
        keepdim: bool = True,
    ) -> None:

        super().__init__()
        self.dim = int(dim)
        self.p = int(p)
        self.keepdim = bool(keepdim)
        self.eps = 1e-10

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        # Lp norm between x1 and x2
        dist = torch.norm(x2 - x1, p=self.p, dim=self.dim, keepdim=self.keepdim)

        return dist


class NeuralDistance(nn.Module):
    """Neural Distance

    Transforms two vectors into a single positive scalar, which can be interpreted as a distance.
    """

    def __init__(self, rep_dim: int, layers: int = 1) -> None:

        super().__init__()

        rep_dim = int(rep_dim)
        layers = int(layers)
        if layers < 1:
            raise ValueError("layers>=1 is required")
        net_features_dim = np.linspace(rep_dim, 1, layers + 1).astype(int)

        net = []
        for i in range(layers):
            net.append(torch.nn.Linear(net_features_dim[i], net_features_dim[i + 1]))
            if i < (layers - 1):
                net.append(torch.nn.ReLU())

        net.append(torch.nn.Softplus(beta=1))

        self.net = torch.nn.Sequential(*net)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:

        out = self.net(x2 - x1)

        return out


class BinaryOnX1(NeuralDistance):
    """Turns Contrast Classifier to a Binary Classifier for x1

    Effectively undo the contrastive approach from inside,
    and transforms the contrast classifier into a conventional binary classifier
    which maps x1 to a single real value representing the
    logits of the positive class.

    The contrast classifier above assumes
    p = 1 - exp( -dist(x1,x2) )
    and returns logits_different = log(p/(1-p))

    Here we define
    dist(x1,x2) = softplus( net(x) ) = log(1+exp(net(x)))

    So we have
    p = 1 - 1/(1+exp(net(x)))
    and so
    log(p/(1-p)) = net(x)
    Therefore, the output of the contrast classifier
    would be effectively net(x)
    """

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:

        out = self.net(x1)

        return out
