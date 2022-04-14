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

from typing import Tuple

import torch
import torch.nn as nn

from gluonts.core.component import validated


class MeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along
    dimension ``dim``, and scales the data accordingly.

    Parameters
    ----------
    dim
        dimension along which to compute the scale
    keepdim
        controls whether to retain dimension ``dim`` (of length 1) in the
        scale tensor, or suppress it.
    minimum_scale
        default scale that is used for elements that are constantly zero
        along dimension ``dim``.
    """

    @validated()
    def __init__(
        self, dim: int, keepdim: bool = False, minimum_scale: float = 1e-10
    ):
        super().__init__()
        assert dim > 0, (
            "Cannot compute scale along dim = 0 (batch dimension), please"
            " provide dim > 0"
        )
        self.dim = dim
        self.keepdim = keepdim
        self.register_buffer("minimum_scale", torch.tensor(minimum_scale))

    def forward(
        self, data: torch.Tensor, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # these will have shape (N, C)
        total_weight = weights.sum(dim=self.dim)
        weighted_sum = (data.abs() * weights).sum(dim=self.dim)

        # first compute a global scale per-dimension
        total_observed = total_weight.sum(dim=0)
        denominator = torch.max(
            total_observed, torch.ones_like(total_observed)
        )
        default_scale = weighted_sum.sum(dim=0) / denominator

        # then compute a per-item, per-dimension scale
        denominator = torch.max(total_weight, torch.ones_like(total_weight))
        scale = weighted_sum / denominator

        # use per-batch scale when no element is observed
        # or when the sequence contains only zeros
        scale = (
            torch.max(
                self.minimum_scale,
                torch.where(
                    weighted_sum > torch.zeros_like(weighted_sum),
                    scale,
                    default_scale * torch.ones_like(total_weight),
                ),
            )
            .detach()
            .unsqueeze(dim=self.dim)
        )

        return data / scale, scale if self.keepdim else scale.squeeze(
            dim=self.dim
        )


class NOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along dimension ``dim``, and therefore
    applies no scaling to the input data.

    Parameters
    ----------
    dim
        dimension along which to compute the scale
    keepdim
        controls whether to retain dimension ``dim`` (of length 1) in the
        scale tensor, or suppress it.
    """

    @validated()
    def __init__(self, dim: int, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = torch.ones_like(data).mean(
            dim=self.dim,
            keepdim=self.keepdim,
        )
        return data, scale
