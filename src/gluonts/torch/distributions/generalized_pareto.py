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
from numbers import Number, Real
from typing import Dict, Tuple

import torch
from torch import nan, inf
import torch.nn.functional as F
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

from .distribution_output import DistributionOutput


class GeneralizedPareto(Distribution):
    r"""
    Creates a Generalized Pareto distribution parameterized by :attr:`loc`, :attr:`scale`, and :attr:`concentration`.

    The Generalized Pareto distribution is a family of continuous probability distributions on the real line.
    Special cases include Exponential (when  :attr:`loc` = 0, :attr:`concentration` = 0), Pareto (when :attr:`concentration` > 0,
     :attr:`loc` = :attr:`scale` / :attr:`concentration`), and Uniform (when :attr:`concentration` = -1).

    This distribution is often used to model the tails of other distributions. This implementation is based on the implementation in TensorFlow Probability.

    Example::

        >>> m = GeneralizedPareto(torch.tensor([0.1]), torch.tensor([2.0]), torch.tensor([0.4]))
        >>> m.sample()  # sample from a Generalized Pareto distribution with loc=1, scale=1, and concentration=1
        tensor([ 1.5623])

    Args:
        loc (float or Tensor): Location parameter of the distribution
        scale (float or Tensor): Scale parameter of the distribution
        concentration (float or Tensor): Concentration parameter of the distribution
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "concentration": constraints.real,
    }
    has_rsample = False

    def __init__(self, loc, scale, concentration, validate_args=None):
        self.loc, self.scale, self.concentration = broadcast_all(
            loc, scale, concentration
        )
        if (
            isinstance(loc, Number)
            and isinstance(scale, Number)
            and isinstance(concentration, Number)
        ):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GeneralizedPareto, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        super(GeneralizedPareto, new).__init__(
            batch_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
            return self.icdf(u)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = self._z(value)
        eq_zero = torch.isclose(self.concentration, torch.tensor(0.0))
        safe_conc = torch.where(
            eq_zero, torch.ones_like(self.concentration), self.concentration
        )
        y = 1 / safe_conc + torch.ones_like(z)
        where_nonzero = torch.where(y == 0, y, y * torch.log1p(safe_conc * z))
        log_scale = (
            math.log(self.scale)
            if isinstance(self.scale, Real)
            else self.scale.log()
        )
        return -log_scale - torch.where(eq_zero, z, where_nonzero)

    def log_survival_function(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = self._z(value)
        eq_zero = torch.isclose(self.concentration, torch.tensor(0.0))
        safe_conc = torch.where(
            eq_zero, torch.ones_like(self.concentration), self.concentration
        )
        where_nonzero = -torch.log1p(safe_conc * z) / safe_conc
        return torch.where(eq_zero, -z, where_nonzero)

    def log_cdf(self, value):
        return torch.log1p(-torch.exp(self.log_survival_function(value)))

    def cdf(self, value):
        return torch.exp(self.log_cdf(value))

    def icdf(self, value):
        loc = self.loc
        scale = self.scale
        concentration = self.concentration
        eq_zero = torch.isclose(concentration, torch.zeros_like(concentration))
        safe_conc = torch.where(
            eq_zero, torch.ones_like(concentration), concentration
        )
        logu = torch.log1p(-value)
        where_nonzero = loc + scale / safe_conc * torch.expm1(
            -safe_conc * logu
        )
        where_zero = loc - scale * logu
        return torch.where(eq_zero, where_zero, where_nonzero)

    def _z(self, x):
        return (x - self.loc) / self.scale

    @property
    def mean(self):
        concentration = self.concentration
        valid = concentration < 1
        safe_conc = torch.where(valid, concentration, 0.5)
        result = self.loc + self.scale / (1 - safe_conc)
        return torch.where(valid, result, torch.full_like(result, nan))

    @property
    def variance(self):
        concentration = self.concentration
        valid = concentration < 0.5
        safe_conc = torch.where(valid, concentration, 0.25)
        result = self.scale**2 / (
            (1 - safe_conc) ** 2 * (1 - 2 * safe_conc)
        ) + torch.zeros_like(self.loc)
        return torch.where(valid, result, torch.full_like(result, nan))

    def entropy(self):
        ans = torch.log(self.scale) + self.concentration + 1
        return torch.broadcast_to(ans, self._batch_shape)

    @property
    def mode(self):
        return self.loc

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        neg_conc = self.concentration < 0
        upper = torch.where(
            neg_conc,
            self.loc - self.scale / self.concentration,
            torch.full_like(self.loc, inf),
        )
        lower = self.loc
        return constraints.interval(lower, upper)


class GeneralizedParetoOutput(DistributionOutput):
    distr_cls: type = GeneralizedPareto
    args_dim: Dict[str, int] = {"loc": 1, "scale": 1, "concentration": 1}

    @classmethod
    def domain_map(
        cls,
        loc: torch.Tensor,
        scale: torch.Tensor,
        concentration: torch.Tensor,
    ):  # type: ignore
        scale = F.softplus(scale)
        # Clamp concentration to avoid numerical issues
        concentration = torch.tanh(concentration)

        # Adjust loc for negative concentration
        neg_conc = concentration < 0
        loc = torch.where(neg_conc, loc - scale / concentration, loc)
        return loc.squeeze(-1), scale.squeeze(-1), concentration.squeeze(-1)

    @property
    def event_shape(self) -> Tuple:
        return ()
