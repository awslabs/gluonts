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
from typing import Tuple, List, Dict

# Third-party imports
import numpy as np

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor
from gluonts.support.util import erfinv

# Relative imports
from .distribution import Distribution, _sample_multiple, getF, softplus
from .distribution_output import DistributionOutput


class LogitNormal(Distribution):
    r"""
    The logit-normal distribution.

    Parameters
    ----------
    mu
        Tensor containing the location, of shape `(*batch_shape, *event_shape)`.
    sigma
        Tensor indicating the scale, of shape `(*batch_shape, *event_shape)`.
    F
    """

    @validated()
    def __init__(self, mu: Tensor, sigma: Tensor, F=None) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma
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
        F = self.F
        x_clip = 1e-3
        x = F.clip(x, x_clip, 1 - x_clip)
        log_prob = -1.0 * (
            F.log(self.sigma)
            + F.log(F.sqrt(2 * F.full(1, np.pi)))
            + F.log(x)
            + F.log(1 - x)
            + (
                (F.log(x) - F.log(1 - x) - self.mu) ** 2
                / (2 * (self.sigma ** 2))
            )
        )
        return log_prob

    def sample(self, num_samples=None, dtype=np.float32):
        def s(mu):
            F = self.F
            q_min = 1e-3
            q_max = 1 - q_min
            sample = F.sample_uniform(
                F.ones_like(mu) * F.full(1, q_min),
                F.ones_like(mu) * F.full(1, q_max),
            )
            transf_sample = self.quantile(sample)
            return transf_sample

        mult_samp = _sample_multiple(s, self.mu, num_samples=num_samples)
        return mult_samp

    def quantile(self, level: Tensor) -> Tensor:
        F = self.F
        exp = F.exp(
            self.mu
            + (self.sigma * F.sqrt(F.full(1, 2)) * erfinv(F, 2 * level - 1))
        )
        return exp / (1 + exp)

    @property
    def args(self) -> List:
        return [self.mu, self.sigma]


class LogitNormalOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"mu": 1, "sigma": 1}
    distr_cls: type = LogitNormal

    @classmethod
    def domain_map(cls, F, mu, sigma):
        sigma = softplus(F, sigma)
        return mu.squeeze(axis=-1), sigma.squeeze(axis=-1)

    @property
    def event_shape(self) -> Tuple:
        return ()

    def distribution(
        self, distr_args, loc=None, scale=None, **kwargs
    ) -> Distribution:
        return self.distr_cls(*distr_args)
