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

from __future__ import annotations
from typing import Optional

import torch

from gluonts.core.component import validated


class Scaler:
    def __call__(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MeanScaler(Scaler):
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
    default_scale
        default scale that is used for elements that are constantly zero
    minimum_scale
        minimum possible scale that is used for any item.
    """

    @validated()
    def __init__(
        self,
        dim: int = -1,
        keepdim: bool = False,
        default_scale: Optional[float] = None,
        minimum_scale: float = 1e-10,
    ) -> None:
        self.dim = dim
        self.keepdim = keepdim
        self.default_scale = default_scale
        self.minimum_scale = minimum_scale

    def __call__(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: (N, [C], T=1)
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        scale = ts_sum / torch.clamp(num_observed, min=1)

        # If `default_scale` is provided, we use it, otherwise we use the scale
        # of the batch.
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # apply default scale where there are no observations
        scale = torch.where(
            num_observed > 0,
            scale,
            default_scale,
        )

        # ensure the scale is at least `self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)

        scaled_data = data / scale

        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        loc = torch.zeros_like(scale)

        return scaled_data, loc, scale


class NOPScaler(Scaler):
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
    def __init__(
        self,
        dim: int = -1,
        keepdim: bool = False,
    ) -> None:
        self.dim = dim
        self.keepdim = keepdim

    def __call__(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = torch.ones_like(data).mean(
            dim=self.dim,
            keepdim=self.keepdim,
        )
        loc = torch.zeros_like(scale)
        return data, loc, scale


class StdScaler(Scaler):
    """
    Computes a std scaling  value along dimension ``dim``, and scales the data accordingly.

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
        self,
        dim: int = -1,
        keepdim: bool = False,
        minimum_scale: float = 1e-5,
    ) -> None:
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale

    def __call__(
        self, data: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            data.shape == weights.shape
        ), "data and weights must have same shape"
        with torch.no_grad():
            denominator = weights.sum(self.dim, keepdim=self.keepdim)
            denominator = denominator.clamp_min(1.0)
            loc = (data * weights).sum(
                self.dim, keepdim=self.keepdim
            ) / denominator

            variance = (((data - loc) * weights) ** 2).sum(
                self.dim, keepdim=self.keepdim
            ) / denominator
            scale = torch.sqrt(variance + self.minimum_scale)
            return (data - loc) / scale, loc, scale
