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

from numbers import Number
from typing import Dict, Optional, Tuple, cast

import numpy as np
import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

from gluonts.core.component import validated

from .distribution_output import DistributionOutput


class GeneralizedPareto(Distribution):
    r"""
    Generalised Pareto distribution.

    Parameters
    ----------
    xi
        Tensor containing the xi (heaviness) shape parameters. The tensor is
        of shape (*batch_shape, 1)
    beta
        Tensor containing the beta scale parameters. The tensor is of
        shape (*batch_shape, 1)
    """
    arg_constraints = {
        "xi": constraints.positive,
        "beta": constraints.positive,
    }
    support = constraints.positive
    has_rsample = False

    def __init__(self, xi, beta, validate_args=None):

        self.xi, self.beta = broadcast_all(
            xi.squeeze(dim=-1), beta.squeeze(dim=-1)
        )

        setattr(self, "xi", xi)
        setattr(self, "beta", beta)

        super(GeneralizedPareto, self).__init__()

        if isinstance(xi, Number) and isinstance(beta, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.xi.size()
        super(GeneralizedPareto, self).__init__(
            batch_shape, validate_args=validate_args
        )

        if (
            self._validate_args
            and not torch.lt(-self.beta, torch.zeros_like(self.beta)).all()
        ):
            raise ValueError("GenPareto is not defined when scale beta<=0")

    @property
    def mean(self):
        """
        Returns the mean of the distribution, of shape (*batch_shape,)
        """
        mu = torch.where(
            self.xi < 1,
            torch.div(self.beta, 1 - self.xi),
            np.nan * torch.ones_like(self.xi),
        )
        return mu

    @property
    def variance(self):
        """
        Returns the variance of the distribution, of shape (*batch_shape,)
        """
        xi, beta = self.xi, self.beta
        var = torch.where(
            xi < 1 / 2.0,
            torch.div(beta**2, torch.mul((1 - xi) ** 2, (1 - 2 * xi))),
            np.nan * torch.ones_like(xi),
        )
        return var

    @property
    def stddev(self):
        return torch.sqrt(self.variance)

    def log_prob(self, x):
        """
        Log probability for a tensor x of shape (*batch_shape)
        """
        # both xi and beta have shape (*batch_shape)
        # and so do all the elements bellow

        x = x.unsqueeze(dim=-1)

        logp = -self.beta.log().double()
        logp += torch.where(
            self.xi == torch.zeros_like(self.xi),
            -x / self.beta,
            -(1 + 1.0 / (self.xi + 1e-6))
            * torch.log(1 + self.xi * x / self.beta),
        )
        logp = torch.where(
            x < torch.zeros_like(x),
            (-np.inf * torch.ones_like(x)).double(),
            logp,
        )
        return logp.squeeze(dim=-1)

    def cdf(self, x):
        """
        cdf values for a tensor x of shape (*batch_shape)
        """
        x = x.unsqueeze(dim=-1)
        x_shifted = torch.div(x, self.beta)
        u = 1 - torch.pow(1 + self.xi * x_shifted, -torch.reciprocal(self.xi))
        return u.squeeze(dim=-1)

    def icdf(self, value):
        """
        icdf values for a tensor quantile values of shape (*batch_shape)
        """
        value = value.unsqueeze(dim=-1)
        x_shifted = torch.div(torch.pow(1 - value, -self.xi) - 1, self.xi)
        x = torch.mul(x_shifted, self.beta)
        return x.squeeze(dim=-1)


class GeneralizedParetoOutput(DistributionOutput):
    distr_cls: type = GeneralizedPareto

    @validated()
    def __init__(
        self,
    ) -> None:
        super().__init__(self)

        self.args_dim = cast(
            Dict[str, int],
            {
                "xi": 1,
                "beta": 1,
            },
        )

    @classmethod
    def domain_map(
        cls,
        xi: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        xi = torch.abs(xi)
        beta = torch.abs(beta)

        return xi, beta

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = 0,
        scale: Optional[torch.Tensor] = None,
    ) -> GeneralizedPareto:
        return self.distr_cls(
            *distr_args,
        )

    @property
    def event_shape(self) -> Tuple:
        return ()
