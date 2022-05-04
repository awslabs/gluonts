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

from typing import Dict, List, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch.distributions import AffineTransform, TransformedDistribution

from gluonts.core.component import validated

from .distribution_output import DistributionOutput


class PiecewiseLinear(torch.distributions.Distribution):
    def __init__(
        self,
        gamma: torch.Tensor,
        slopes: torch.Tensor,
        knot_spacings: torch.Tensor,
        validate_args=False,
    ) -> None:
        self.gamma, self.slopes, self.knot_spacings = (
            gamma,
            slopes,
            knot_spacings,
        )
        self.b, self.knot_positions = PiecewiseLinear._to_orig_params(
            slopes, knot_spacings
        )

        # self.batch_shape = self.gamma.shape
        super().__init__(
            batch_shape=self.batch_shape, validate_args=validate_args
        )

    @staticmethod
    def _to_orig_params(
        slopes: torch.Tensor, knot_spacings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = PiecewiseLinear.parametrize_slopes(slopes)
        knot_positions = PiecewiseLinear.parametrize_knots(knot_spacings)
        return b, knot_positions

    @staticmethod
    def parametrize_slopes(slopes: torch.Tensor) -> torch.Tensor:
        slopes_parametrized = torch.diff(slopes, dim=-1)
        slopes_parametrized = torch.cat(
            [slopes[..., 0:1], slopes_parametrized], dim=-1
        )

        return slopes_parametrized

    @staticmethod
    def parametrize_knots(knot_spacings: torch.Tensor) -> torch.Tensor:
        # the last entry is 1 and can be omitted
        knot_positions = torch.cumsum(knot_spacings, dim=-1)[..., :-1]

        # the first knot pos is 0
        knot_positions = torch.cat(
            [torch.zeros_like(knot_positions[..., 0:1]), knot_positions],
            dim=-1,
        )

        return knot_positions

    def quantile_internal(
        self, u: torch.Tensor, dim: Optional[int] = None
    ) -> torch.Tensor:
        # output shape = u.shape
        if dim is not None:
            # when num_samples!=None
            #
            # In testing
            # dim = 0
            # u.shape          = (num_samples, *batch_shape)
            # gamma.shape      = (1, *batch_shape)
            # other_para.shape = (1, *batch_shape, num_pieces)
            #
            # In training, u_tilde is needed to compute CRPS
            # dim = -2
            # u.shape          = (*batch_shape, num_pieces)
            # gamma.shape      = (*batch_shape, 1)
            # other_para.shape = (*batch_shape, 1, num_pieces)

            gamma = self.gamma.unsqueeze(dim=0 if dim == 0 else -1)
            knot_positions, b = (
                self.knot_positions.unsqueeze(dim=dim),
                self.b.unsqueeze(dim=dim),
            )
        else:
            # when num_sample is None
            #
            # In testing
            # dim = None
            # u.shape          = (*batch_shape)
            # gamma.shape      = (*batch_shape)
            # other_para.shape = (*batch_shape, num_pieces)

            gamma = self.gamma
            knot_positions, b = self.knot_positions, self.b

        u = u.unsqueeze(-1)
        u_spline = F.relu(u - knot_positions)
        quantile = gamma + torch.sum(b * u_spline, dim=-1)

        return quantile

    def quantile(self, u: torch.Tensor) -> torch.Tensor:
        return self.quantile_internal(u, dim=0)

    def cdf(self, z: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma
        b, knot_positions = self.b, self.knot_positions

        knots_eval = self.quantile_internal(knot_positions, dim=-2)
        mask = torch.lt(knots_eval, z.unsqueeze(-1))

        sum_slopes = torch.sum(mask * b, dim=-1)

        zero_val = torch.zeros(
            1, dtype=gamma.dtype, device=gamma.device, layout=gamma.layout
        )
        one_val = torch.ones(
            1, dtype=gamma.dtype, device=gamma.device, layout=gamma.layout
        )

        sum_slopes_nz = torch.where(
            sum_slopes == zero_val, one_val, sum_slopes
        )

        u_tilde = torch.where(
            sum_slopes == zero_val,
            zero_val,
            (z - gamma + torch.sum(b * knot_positions * mask, dim=-1))
            / sum_slopes_nz,
        )

        u_tilde = torch.min(u_tilde, one_val)

        return u_tilde

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        return self.crps(z)

    def crps(self, z: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma
        b, knot_positions = self.b, self.knot_positions

        u_tilde = self.cdf(z)

        max_u_tilde_knots = torch.max(u_tilde.unsqueeze(-1), knot_positions)

        coeff = (
            (1 - knot_positions**3) / 3
            - knot_positions
            - max_u_tilde_knots**2
            + 2 * max_u_tilde_knots * knot_positions
        )

        result = (2 * u_tilde - 1) * (z - gamma) + torch.sum(b * coeff, dim=-1)

        return result

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        target_shape = (
            self.gamma.shape
            if sample_shape == torch.Size()
            else torch.Size(sample_shape) + self.gamma.shape
        )

        u = torch.rand(
            target_shape,
            dtype=self.gamma.dtype,
            device=self.gamma.device,
            layout=self.gamma.layout,
        )

        sample = self.quantile(u)

        if sample_shape == torch.Size():
            sample = sample.squeeze(dim=0)

        return sample

    @property
    def batch_shape(self) -> torch.Size():
        return self.gamma.shape


class PiecewiseLinearOutput(DistributionOutput):
    distr_cls: type = PiecewiseLinear

    @validated()
    def __init__(self, num_pieces: int) -> None:
        super().__init__(self)

        assert (
            isinstance(num_pieces, int) and num_pieces > 1
        ), "num_pieces should be an integer and greater than 1"

        self.num_pieces = num_pieces
        self.args_dim = cast(
            Dict[str, int],
            {"gamma": 1, "slopes": num_pieces, "knot_spacings": num_pieces},
        )

    @classmethod
    def domain_map(
        cls,
        gamma: torch.Tensor,
        slopes: torch.Tensor,
        knot_spacings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        slopes_nn = torch.abs(slopes)

        knot_spacings_proj = F.softmax(knot_spacings, dim=-1)

        return gamma.squeeze(dim=-1), slopes_nn, knot_spacings_proj

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = 0,
        scale: Optional[torch.Tensor] = None,
    ) -> PiecewiseLinear:
        if scale is None:
            return self.distr_cls(*distr_args)
        else:
            distr = self.distr_cls(*distr_args)
            return TransformedPiecewiseLinear(
                distr, [AffineTransform(loc=loc, scale=scale)]
            )

    @property
    def event_shape(self) -> Tuple:
        return ()


class TransformedPiecewiseLinear(TransformedDistribution):
    @validated()
    def __init__(
        self,
        base_distribution: PiecewiseLinear,
        transforms: List[AffineTransform],
        validate_args=None,
    ) -> None:
        super().__init__(
            base_distribution, transforms, validate_args=validate_args
        )

    def crps(self, y: torch.Tensor) -> torch.Tensor:
        z = y
        scale = 1.0
        for t in self.transforms[::-1]:
            assert isinstance(t, AffineTransform), "Not an AffineTransform"
            z = t._inverse(y)
            scale *= t.scale
        p = self.base_dist.crps(z)
        return p * scale
