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

# mainly based from https://github.com/pytorch/rl/blob/main/torchrl/modules/distributions/continuous.py#L166

from numbers import Number
from typing import Dict, Optional, Tuple, Union

import torch
from torch import distributions as D
from torch.distributions import constraints

import torch.nn.functional as F
from .distribution_output import DistributionOutput
from gluonts.core.component import validated
from torch.distributions import Distribution
from .utils.truncated_normal import TruncatedNormal as _TruncatedNormal

# speeds up distribution construction
D.Distribution.set_default_validate_args(False)


class TruncatedNormal(D.Independent):
    """Implements a Truncated Normal distribution with location scaling.

    Location scaling prevents the location to be "too far" from 0, which ultimately
    leads to numerically unstable samples and poor gradient computation (e.g. gradient explosion).
    In practice, the location is computed according to

        .. math::
            loc = tanh(loc / upscale) * upscale.

    This behaviour can be disabled by switching off the tanh_loc parameter (see below).


    Args:
        loc (torch.Tensor): normal distribution location parameter
        scale (torch.Tensor): normal distribution sigma parameter (squared root of variance)
        upscale (torch.Tensor or number, optional): 'a' scaling factor in the formula:

            .. math::
                loc = tanh(loc / upscale) * upscale.

            Default is 5.0

        min (torch.Tensor or number, optional): minimum value of the distribution. Default = -1.0;
        max (torch.Tensor or number, optional): maximum value of the distribution. Default = 1.0;
        tanh_loc (bool, optional): if ``True``, the above formula is used for
            the location scaling, otherwise the raw value is kept.
            Default is ``False``;
    """

    num_params: int = 2

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.greater_than(1e-6),
    }

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        upscale: Union[torch.Tensor, float] = 5.0,
        min: Union[torch.Tensor, float] = -1.0,
        max: Union[torch.Tensor, float] = 1.0,
        tanh_loc: bool = False,
    ):
        err_msg = (
            "TanhNormal max values must be strictly greater than min values"
        )
        if isinstance(max, torch.Tensor) or isinstance(min, torch.Tensor):
            if not (max > min).all():
                raise RuntimeError(err_msg)
        elif isinstance(max, Number) and isinstance(min, Number):
            if not max > min:
                raise RuntimeError(err_msg)
        else:
            if not all(max > min):
                raise RuntimeError(err_msg)

        if isinstance(max, torch.Tensor):
            self.non_trivial_max = (max != 1.0).any()
        else:
            self.non_trivial_max = max != 1.0

        if isinstance(min, torch.Tensor):
            self.non_trivial_min = (min != -1.0).any()
        else:
            self.non_trivial_min = min != -1.0
        self.tanh_loc = tanh_loc

        self.device = loc.device
        self.upscale = (
            upscale
            if not isinstance(upscale, torch.Tensor)
            else upscale.to(self.device)
        )

        if isinstance(max, torch.Tensor):
            max = max.to(self.device)
        else:
            max = torch.tensor(max, device=self.device)
        if isinstance(min, torch.Tensor):
            min = min.to(self.device)
        else:
            min = torch.tensor(min, device=self.device)
        self.min = min
        self.max = max
        self.update(loc, scale)

    def update(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        if self.tanh_loc:
            loc = (loc / self.upscale).tanh() * self.upscale
        if self.non_trivial_max or self.non_trivial_min:
            loc = loc + (self.max - self.min) / 2 + self.min
        self.loc = loc
        self.scale = scale

        base_dist = _TruncatedNormal(
            loc, scale, self.min.expand_as(loc), self.max.expand_as(scale)
        )
        super().__init__(base_dist, 1, validate_args=False)

    @property
    def mode(self):
        m = self.base_dist.loc
        a = self.base_dist._non_std_a + self.base_dist._dtype_min_gt_0
        b = self.base_dist._non_std_b - self.base_dist._dtype_min_gt_0
        m = torch.min(torch.stack([m, b], -1), dim=-1)[0]
        return torch.max(torch.stack([m, a], -1), dim=-1)[0]

    def log_prob(self, value, **kwargs):
        a = self.base_dist._non_std_a + self.base_dist._dtype_min_gt_0
        a = a.expand_as(value)
        b = self.base_dist._non_std_b - self.base_dist._dtype_min_gt_0
        b = b.expand_as(value)
        value = torch.min(torch.stack([value, b], -1), dim=-1)[0]
        value = torch.max(torch.stack([value, a], -1), dim=-1)[0]
        return self.base_dist.log_prob(
            value
        )  # original: return super().log_prob(value, **kwargs)


class TruncatedNormalOutput(DistributionOutput):
    distr_cls: type = TruncatedNormal

    @validated()
    def __init__(
        self,
        min: float,
        max: float,
        upscale: float = 5.0,
        tanh_loc: bool = False,
    ) -> None:
        super().__init__(self)

        self.min = min
        self.max = max
        self.upscale = upscale
        self.tanh_loc = tanh_loc
        self.args_dim: Dict[str, int] = {
            "loc": 1,
            "scale": 1,
        }

    # @classmethod
    def domain_map(
        self,
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
