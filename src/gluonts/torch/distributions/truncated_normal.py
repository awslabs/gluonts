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

import math
from numbers import Number
from typing import Dict, Optional, Tuple, Union

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

import torch.nn.functional as F
from .distribution_output import DistributionOutput
from gluonts.core.component import validated
from torch.distributions import Distribution

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedNormal(Distribution):
    """
    Implements a Truncated Normal distribution with location scaling.

    Location scaling prevents the location to be "too far" from 0, which ultimately
    leads to numerically unstable samples and poor gradient computation (e.g. gradient explosion).
    In practice, the location is computed according to

        .. math::
            loc = tanh(loc / upscale) * upscale.

    This behaviour can be disabled by switching off the tanh_loc parameter (see below).

    Parameters
    ----------
    loc:
        normal distribution location parameter
    scale:
        normal distribution sigma parameter (squared root of variance)
    min:
        minimum value of the distribution. Default = -1.0
    max:
        maximum value of the distribution. Default = 1.0
    upscale:
        scaling factor. Default = 5.0
    tanh_loc:
        if ``True``, the above formula is used for
        the location scaling, otherwise the raw value is kept.
        Default is ``False``

    References
    ----------
    - https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

    Notes
    -----
    This implementation is strongly based on:
        - https://github.com/pytorch/rl/blob/main/torchrl/modules/distributions/truncated_normal.py
        - https://github.com/toshas/torch_truncnorm
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.greater_than(1e-6),
    }

    has_rsample = True
    eps = 1e-6

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        min: Union[torch.Tensor, float] = -1.0,
        max: Union[torch.Tensor, float] = 1.0,
        upscale: Union[torch.Tensor, float] = 5.0,
        tanh_loc: bool = False,
    ):
        scale = scale.clamp_min(self.eps)
        if tanh_loc:
            loc = (loc / upscale).tanh() * upscale
        loc = loc + (max - min) / 2 + min

        self.min = min
        self.max = max
        self.upscale = upscale
        self.loc, self.scale, a, b = broadcast_all(
            loc, scale, self.min, self.max
        )
        self._non_std_a = a
        self._non_std_b = b
        self.a = (a - self.loc) / self.scale
        self.b = (b - self.loc) / self.scale

        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()

        super(TruncatedNormal, self).__init__(batch_shape)
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        if any(
            (self.a >= self.b)
            .view(
                -1,
            )
            .tolist()
        ):
            raise ValueError("Incorrect truncation range")

        eps = self.eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp(eps, 1 - eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (
            self._little_phi_b * little_phi_coeff_b
            - self._little_phi_a * little_phi_coeff_a
        ) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = (
            1
            - self._lpbb_m_lpaa_d_Z
            - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        )
        self._entropy = (
            CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z
        )
        self._log_scale = self.scale.log()
        self._mean_non_std = self._mean * self.scale + self.loc
        self._variance_non_std = self._variance * self.scale**2
        self._entropy_non_std = self._entropy + self._log_scale

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean_non_std

    @property
    def variance(self):
        return self._variance_non_std

    @property
    def entropy(self):
        return self._entropy_non_std

    @staticmethod
    def _little_phi(x):
        return (-(x**2) * 0.5).exp() * CONST_INV_SQRT_2PI

    def _big_phi(self, x):
        phi = 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())
        return phi.clamp(self.eps, 1 - self.eps)

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf_truncated_standard_normal(self, value):
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf_truncated_standard_normal(self, value):
        y = self._big_phi_a + value * self._Z
        y = y.clamp(self.eps, 1 - self.eps)
        return self._inv_big_phi(y)

    def log_prob_truncated_standard_normal(self, value):
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value**2) * 0.5

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return self.cdf_truncated_standard_normal(self._to_std_rv(value))

    def icdf(self, value):
        sample = self._from_std_rv(self.icdf_truncated_standard_normal(value))

        # clamp data but keep gradients
        sample_clip = torch.stack(
            [sample.detach(), self._non_std_a.detach().expand_as(sample)], 0
        ).max(0)[0]
        sample_clip = torch.stack(
            [sample_clip, self._non_std_b.detach().expand_as(sample)], 0
        ).min(0)[0]
        sample.data.copy_(sample_clip)
        return sample

    def log_prob(self, value):
        a = self._non_std_a + self._dtype_min_gt_0
        a = a.expand_as(value)
        b = self._non_std_b - self._dtype_min_gt_0
        b = b.expand_as(value)
        value = torch.min(torch.stack([value, b], -1), dim=-1)[0]
        value = torch.max(torch.stack([value, a], -1), dim=-1)[0]
        value = self._to_std_rv(value)
        return self.log_prob_truncated_standard_normal(value) - self._log_scale

    def rsample(self, sample_shape=None):
        if sample_shape is None:
            sample_shape = torch.Size([])
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(
            self._dtype_min_gt_0, self._dtype_max_lt_1
        )
        return self.icdf(p)


class TruncatedNormalOutput(DistributionOutput):
    distr_cls: type = TruncatedNormal

    @validated()
    def __init__(
        self,
        min: float = -1.0,
        max: float = 1.0,
        upscale: float = 5.0,
        tanh_loc: bool = False,
    ) -> None:
        assert min < max, "max must be strictly greater than min"

        super().__init__(self)

        self.min = min
        self.max = max
        self.upscale = upscale
        self.tanh_loc = tanh_loc
        self.args_dim: Dict[str, int] = {
            "loc": 1,
            "scale": 1,
        }

    @classmethod
    def domain_map(  # type: ignore
        cls,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ):
        scale = F.softplus(scale)

        return (
            loc.squeeze(-1),
            scale.squeeze(-1),
        )

    # Overwrites the parent class method: We pass constant float and
    # boolean parameters across tensors
    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        (loc, scale) = distr_args

        assert isinstance(loc, torch.Tensor)
        assert isinstance(scale, torch.Tensor)

        return TruncatedNormal(
            loc=loc,
            scale=scale,
            upscale=self.upscale,
            min=self.min,
            max=self.max,
            tanh_loc=self.tanh_loc,
        )

    @property
    def event_shape(self) -> Tuple:
        return ()
