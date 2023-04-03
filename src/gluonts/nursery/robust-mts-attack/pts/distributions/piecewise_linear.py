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

import torch
import torch.nn.functional as F
from torch.distributions import (
    constraints,
    NegativeBinomial,
    Poisson,
    Distribution,
    TransformedDistribution,
    AffineTransform,
)
from torch.distributions.utils import broadcast_all, lazy_property

from .utils import broadcast_shape


class PiecewiseLinear(Distribution):
    def __init__(self, gamma, slopes, knot_spacings, validate_args=None):
        self.gamma = gamma
        self.slopes = slopes
        self.knot_spacings = knot_spacings

        self.b, self.knot_positions = PiecewiseLinear._to_orig_params(
            slopes=slopes, knot_spacings=knot_spacings
        )
        super(PiecewiseLinear, self).__init__(
            batch_shape=self.gamma.shape, validate_args=validate_args
        )

    @staticmethod
    def _to_orig_params(slopes, knot_spacings):
        # b: the difference between slopes of consecutive pieces
        b = slopes[..., 1:] - slopes[..., 0:-1]

        # Add slope of first piece to b: b_0 = m_0
        m_0 = slopes[..., 0:1]
        b = torch.cat((m_0, b), dim=-1)

        # The actual position of the knots is obtained by cumulative sum of
        # the knot spacings. The first knot position is always 0 for quantile
        # functions.
        knot_positions = torch.cumsum(knot_spacings, dim=-1) - knot_spacings

        return b, knot_positions

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand_like(self.gamma.expand(shape))

        sample = self.quantile(u)

        if len(sample_shape) == 0:
            sample = sample.squeeze(0)

        return sample

    def quantile(self, level):
        return self.quantile_internal(level, dim=0)

    def quantile_internal(self, x, dim=None):
        if dim is not None:
            gamma = self.gamma.unsqueeze(dim=dim if dim == 0 else -1)
            knot_positions = self.knot_positions.unsqueeze(dim)
            b = self.b.unsqueeze(dim)
        else:
            gamma, knot_positions, b = self.gamma, self.knot_positions, self.b

        x_minus_knots = x.unsqueeze(-1) - knot_positions

        quantile = gamma + (b * F.relu(x_minus_knots)).sum(-1)

        return quantile

    def log_prob(self, value):
        return -self.crps(value)

    def cdf(self, x):
        gamma, b, knot_positions = self.gamma, self.b, self.knot_positions

        quantiles_at_knots = self.quantile_internal(knot_positions, dim=-2)

        # Mask to nullify the terms corresponding to knots larger than l_0,
        # which is the largest knot (quantile level) such that the quantile
        # at l_0, s(l_0) < x.
        mask = torch.le(quantiles_at_knots, x.unsqueeze(-1))

        slope_l0 = (b * mask).sum(-1)

        # slope_l0 can be zero in which case a_tilde = 0.
        # The following is to circumvent an issue where the
        # backward() returns nans when slope_l0 is zero in the where
        slope_l0_nz = torch.where(
            slope_l0 == 0.0, torch.ones_like(x), slope_l0
        )

        a_tilde = torch.where(
            slope_l0 == 0.0,
            torch.zeros_like(x),
            (x - gamma + (b * knot_positions * mask).sum(-1)) / slope_l0_nz,
        )

        return torch.clamp(a_tilde, min=0.0, max=1.0)

    def crps(self, x):
        gamma, b, knot_positions = self.gamma, self.b, self.knot_positions

        a_tilde = self.cdf(x)

        max_a_tilde_knots = torch.max(a_tilde.unsqueeze(-1), knot_positions)

        knots_cubed = torch.pow(knot_positions, 3.0)
        coeff = (
            (1.0 - knots_cubed) / 3.0
            - knot_positions
            - torch.square(max_a_tilde_knots)
            + 2 * max_a_tilde_knots * knot_positions
        )

        return (
            (2 * a_tilde - 1) * x
            + (1 - 2 * a_tilde) * gamma
            + (b * coeff).sum(-1)
        )


class TransformedPiecewiseLinear(TransformedDistribution):
    def __init__(self, base_distribution, transforms):
        super().__init__(base_distribution, transforms)

    def crps(self, x):
        scale = 1.0

        for transform in reversed(self.transforms):
            assert isinstance(
                transform, AffineTransform
            ), "Not an AffineTransform"
            x = transform.inv(x)
            scale *= transform.scale

        p = self.base_dist.crps(x)
        return p * scale
