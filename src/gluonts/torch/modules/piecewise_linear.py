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

from typing import Dict, Optional, Tuple, List, cast

import torch
import torch.nn.functional as F

from gluonts.core.component import validated
from gluonts.torch.modules.distribution_output import (
    Distribution,
    DistributionOutput,
)

from torch.distributions import (
    AffineTransform,
    TransformedDistribution,
)


class PiecewiseLinear(Distribution):
    def __init__(
        self,
        gamma: torch.Tensor,
        slopes: torch.Tensor,
        knots: torch.Tensor,
        validate_args=False,
    ) -> None:
        self.gamma, self.slopes, self.knots = gamma, slopes, knots
        self.m, self.knots_pos = PiecewiseLinear._to_orig_params(slopes, knots)

        batch_shape = self.gamma.shape
        super(PiecewiseLinear, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    @staticmethod
    def _to_orig_params(
        slopes: torch.Tensor, knots: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = PiecewiseLinear.parametrize_slopes(slopes)
        knots_pos = PiecewiseLinear.parametrize_knots(knots)
        return m, knots_pos

    @staticmethod
    def parametrize_slopes(slopes: torch.Tensor) -> torch.Tensor:
        slopes_parametrized = slopes
        slopes_parametrized[..., 1:] = torch.diff(slopes, dim=-1)
        return slopes_parametrized

    @staticmethod
    def parametrize_knots(knots: torch.Tensor) -> torch.Tensor:
        knots_pos = torch.cumsum(knots, dim=-1)[
            ..., :-1
        ]  # the last entry is 1 and can be omitted
        return knots_pos

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
            # other_para.shape = (1, *batch_shape, num_knots)
            #
            # In training, u_tilde is needed to compute CRPS
            # dim = -2
            # u.shape          = (*batch_shape, num_knots)
            # gamma.shape      = (*batch_shape, 1)
            # other_para.shape = (*batch_shape, 1, num_knots)

            gamma = self.gamma.unsqueeze(dim=0 if dim == 0 else -1)
            knots_pos, m = self.knots_pos.unsqueeze(dim=dim), self.m.unsqueeze(
                dim=dim
            )
        else:
            # when num_sample is None
            #
            # In testing
            # dim = None
            # u.shape          = (*batch_shape)
            # gamma.shape      = (*batch_shape)
            # other_para.shape = (*batch_shape, num_knots)

            gamma = self.gamma
            knots_pos, m = self.knots_pos, self.m

        u = u.unsqueeze(-1)
        u_spline = F.relu(u - knots_pos)
        quantile = gamma + torch.sum(m * u_spline, dim=-1)

        return quantile

    def quantile(self, u: torch.Tensor) -> torch.Tensor:
        return self.quantile_internal(u, dim=0)

    def get_u_tilde(self, z: torch.Tensor) -> torch.Tensor:
        """
        compute the quantile levels u_tilde s.t. quantile(u_tilde)=z

        Input
        z: observations, shape = gamma.shape = (*batch_size)

        Output
        u_tilde: of type torch.Tensor
        """
        gamma = self.gamma
        m, knots_pos = self.m, self.knots_pos

        knots_eval = self.quantile_internal(knots_pos, dim=-2)
        mask = torch.lt(knots_eval, z.unsqueeze(-1))

        sum_slopes = torch.sum(mask * m, dim=-1)

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
            (z - gamma + torch.sum(m * knots_pos * mask, dim=-1))
            / sum_slopes_nz,
        )

        u_tilde = torch.min(u_tilde, one_val)

        return u_tilde

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        return self.crps(z)

    def crps(self, z: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma
        m, knots_pos = self.m, self.knots_pos

        u_tilde = self.get_u_tilde(z)

        max_u_tilde_knots = torch.max(u_tilde.unsqueeze(-1), knots_pos)

        coeff = (
            (1 - knots_pos ** 3) / 3
            - knots_pos
            - max_u_tilde_knots ** 2
            + 2 * max_u_tilde_knots * knots_pos
        )

        result = (2 * u_tilde - 1) * (z - gamma) + torch.sum(m * coeff, dim=-1)

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
            {"gamma": 1, "slopes": num_pieces, "knots": num_pieces},
        )

    @classmethod
    def domain_map(
        cls, gamma: torch.Tensor, slopes: torch.Tensor, knots: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        slopes_nn = torch.abs(slopes)

        knots = F.softmax(knots, dim=-1)
        knots = torch.cat([torch.zeros_like(knots[..., 0:1]), knots], dim=-1)

        return gamma.squeeze(dim=-1), slopes_nn, knots

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
