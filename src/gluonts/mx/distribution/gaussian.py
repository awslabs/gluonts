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
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .distribution import Distribution, _sample_multiple, getF, softplus
from .distribution_output import DistributionOutput


class Gaussian(Distribution):
    r"""
    Gaussian distribution.

    Parameters
    ----------
    mu
        Tensor containing the means, of shape `(*batch_shape, *event_shape)`.
    std
        Tensor containing the standard deviations, of shape
        `(*batch_shape, *event_shape)`.
    F
    """

    is_reparameterizable = True

    @validated()
    def __init__(self, mu: Tensor, sigma: Tensor) -> None:
        self.mu = mu
        self.sigma = sigma

    @property
    def F(self):
        return getF(self.mu)

    @property
    def batch_shape(self) -> Tuple:
        return self.mu.shape

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    def log_prob(self, x: Tensor) -> Tensor:
        F = self.F
        mu, sigma = self.mu, self.sigma
        return -1.0 * (
            F.log(sigma)
            + 0.5 * math.log(2 * math.pi)
            + 0.5 * F.square((x - mu) / sigma)
        )

    @property
    def mean(self) -> Tensor:
        return self.mu

    @property
    def stddev(self) -> Tensor:
        return self.sigma

    def cdf(self, x):
        F = self.F
        u = F.broadcast_div(
            F.broadcast_minus(x, self.mu), self.sigma * math.sqrt(2.0)
        )
        return (F.erf(u) + 1.0) / 2.0

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        return _sample_multiple(
            partial(self.F.sample_normal, dtype=dtype),
            mu=self.mu,
            sigma=self.sigma,
            num_samples=num_samples,
        )

    def sample_rep(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        def s(mu: Tensor, sigma: Tensor) -> Tensor:
            raw_samples = self.F.sample_normal(
                mu=mu.zeros_like(), sigma=sigma.ones_like(), dtype=dtype
            )
            return sigma * raw_samples + mu

        return _sample_multiple(
            s, mu=self.mu, sigma=self.sigma, num_samples=num_samples
        )

    def quantile(self, level: Tensor) -> Tensor:
        F = self.F
        # we consider level to be an independent axis and so expand it
        # to shape (num_levels, 1, 1, ...)
        for _ in range(self.all_dim):
            level = level.expand_dims(axis=-1)

        return F.broadcast_add(
            self.mu,
            F.broadcast_mul(
                self.sigma, math.sqrt(2.0) * F.erfinv(2.0 * level - 1.0)
            ),
        )

    @property
    def args(self) -> List:
        return [self.mu, self.sigma]


class GaussianOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"mu": 1, "sigma": 1}
    distr_cls: type = Gaussian

    @classmethod
    def domain_map(cls, F, mu, sigma):
        r"""
        Maps raw tensors to valid arguments for constructing a Gaussian
        distribution.

        Parameters
        ----------
        F
        mu
            Tensor of shape `(*batch_shape, 1)`
        sigma
            Tensor of shape `(*batch_shape, 1)`

        Returns
        -------
        Tuple[Tensor, Tensor]
            Two squeezed tensors, of shape `(*batch_shape)`: the first has the
            same entries as `mu` and the second has entries mapped to the
            positive orthant.
        """
        sigma = softplus(F, sigma)
        return mu.squeeze(axis=-1), sigma.squeeze(axis=-1)

    @property
    def event_shape(self) -> Tuple:
        return ()
