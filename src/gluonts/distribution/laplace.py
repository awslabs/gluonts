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

# Standard library imports
from typing import Dict, Tuple, List

# Third-party imports
import numpy as np

# First-party imports
from gluonts.model.common import Tensor
from gluonts.core.component import validated

# Relative imports
from .distribution import Distribution, _sample_multiple, getF, softplus
from .distribution_output import DistributionOutput


class Laplace(Distribution):
    r"""
    Laplace distribution.

    Parameters
    ----------
    mu
        Tensor containing the means, of shape `(*batch_shape, *event_shape)`.
    b
        Tensor containing the distribution scale, of shape
        `(*batch_shape, *event_shape)`.
    F
    """

    is_reparameterizable = True

    @validated()
    def __init__(self, mu: Tensor, b: Tensor, F=None) -> None:
        self.mu = mu
        self.b = b
        self.F = F if F else getF(mu)

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
        return -1.0 * (
            self.F.log(2.0 * self.b) + self.F.abs((x - self.mu) / self.b)
        )

    @property
    def mean(self) -> Tensor:
        return self.mu

    @property
    def stddev(self) -> Tensor:
        return 2.0 ** 0.5 * self.b

    def cdf(self, x: Tensor) -> Tensor:
        y = (x - self.mu) / self.b
        return 0.5 + 0.5 * y.sign() * (1.0 - self.F.exp(-y.abs()))

    def sample_rep(self, num_samples=None, dtype=np.float32) -> Tensor:
        F = self.F

        def s(mu: Tensor, b: Tensor) -> Tensor:
            ones = mu.ones_like()
            x = F.random.uniform(-0.5 * ones, 0.5 * ones, dtype=dtype)
            laplace_samples = mu - b * F.sign(x) * F.log(
                (1.0 - 2.0 * F.abs(x)).clip(1.0e-30, 1.0e30)
                # 1.0 - 2.0 * F.abs(x)
            )
            return laplace_samples

        return _sample_multiple(
            s, mu=self.mu, b=self.b, num_samples=num_samples
        )

    def quantile(self, level: Tensor) -> Tensor:
        F = self.F
        for _ in range(self.all_dim):
            level = level.expand_dims(axis=-1)

        condition = F.broadcast_greater(level, level.zeros_like() + 0.5)
        u = F.where(condition, F.log(2.0 * level), -F.log(2.0 - 2.0 * level))

        return F.broadcast_add(self.mu, F.broadcast_mul(self.b, u))

    @property
    def args(self) -> List:
        return [self.mu, self.b]


class LaplaceOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"mu": 1, "b": 1}
    distr_cls: type = Laplace

    @classmethod
    def domain_map(cls, F, mu, b):
        b = softplus(F, b)
        return mu.squeeze(axis=-1), b.squeeze(axis=-1)

    @property
    def event_shape(self) -> Tuple:
        return ()
