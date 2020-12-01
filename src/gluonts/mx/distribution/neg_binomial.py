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

from typing import Dict, List, Optional, Tuple

import numpy as np

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .deterministic import DeterministicOutput
from .distribution import Distribution, _sample_multiple, getF, softplus
from .distribution_output import DistributionOutput
from .mixture import MixtureDistributionOutput


class NegativeBinomial(Distribution):
    r"""
    Negative binomial distribution, i.e. the distribution of the number of
    successes in a sequence of independent Bernoulli trials.

    Parameters
    ----------
    mu
        Tensor containing the means, of shape `(*batch_shape, *event_shape)`.
    alpha
        Tensor of the shape parameters, of shape `(*batch_shape, *event_shape)`.
    F
    """

    is_reparameterizable = False

    @validated()
    def __init__(self, mu: Tensor, alpha: Tensor) -> None:
        self.mu = mu
        self.alpha = alpha

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
        alphaInv = 1.0 / self.alpha
        alpha_times_mu = self.alpha * self.mu
        F = self.F
        ll = (
            x * F.log(alpha_times_mu / (1.0 + alpha_times_mu))
            - alphaInv * F.log1p(alpha_times_mu)
            + F.gammaln(x + alphaInv)
            - F.gammaln(x + 1.0)
            - F.gammaln(alphaInv)
        )
        return ll

    @property
    def mean(self) -> Tensor:
        return self.mu

    @property
    def stddev(self) -> Tensor:
        return self.F.sqrt(self.mu * (1.0 + self.mu * self.alpha))

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        def s(mu: Tensor, alpha: Tensor) -> Tensor:
            F = self.F
            tol = 1e-5
            r = 1.0 / alpha
            theta = alpha * mu
            r = F.minimum(F.maximum(tol, r), 1e10)
            theta = F.minimum(F.maximum(tol, theta), 1e10)
            x = F.minimum(F.random.gamma(r, theta), 1e6)
            return F.random.poisson(lam=x, dtype=dtype)

        return _sample_multiple(
            s, mu=self.mu, alpha=self.alpha, num_samples=num_samples
        )

    @property
    def args(self) -> List:
        return [self.mu, self.alpha]


class NegativeBinomialOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"mu": 1, "alpha": 1}
    distr_cls: type = NegativeBinomial

    @classmethod
    def domain_map(cls, F, mu, alpha):
        epsilon = np.finfo(cls._dtype).eps  # machine epsilon

        mu = softplus(F, mu) + epsilon
        alpha = softplus(F, alpha) + epsilon
        return mu.squeeze(axis=-1), alpha.squeeze(axis=-1)

    # Overwrites the parent class method.
    # We cannot scale using the affine transformation since negative binomial should return integers.
    # Instead we scale the parameters.
    def distribution(
        self,
        distr_args,
        loc: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ) -> NegativeBinomial:
        mu, alpha = distr_args
        if scale is None:
            return NegativeBinomial(mu, alpha)
        else:
            F = getF(mu)
            mu = F.broadcast_mul(mu, scale)
            return NegativeBinomial(mu, alpha, F)

    @property
    def event_shape(self) -> Tuple:
        return ()


class ZeroInflatedNegativeBinomialOutput(MixtureDistributionOutput):
    def __init__(self):
        super().__init__([NegativeBinomialOutput(), DeterministicOutput(0)])
