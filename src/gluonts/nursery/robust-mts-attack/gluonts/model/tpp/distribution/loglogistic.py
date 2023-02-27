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

from typing import Dict, Optional, Tuple

import numpy as np
from mxnet import autograd, nd

from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.distribution.distribution import getF

from .base import TPPDistribution, TPPDistributionOutput


class Loglogistic(TPPDistribution):
    r"""
    Log-logistic distribution.

    A very heavy-tailed distribution over positive real numbers.
    https://en.wikipedia.org/wiki/Log-logistic_distribution

    Drawing :math:`x \sim \operatorname{Loglogistic}(\mu, \sigma)` is equivalent
    to:

    .. math::

        y &\sim \operatorname{Logistic}(\mu, \sigma)\\
        x &= \exp(y)
    """

    is_reparametrizable = True

    @validated()
    def __init__(self, mu: Tensor, sigma: Tensor) -> None:
        self.mu = mu
        self.sigma = sigma

    @property
    def batch_shape(self) -> Tuple:
        return self.mu.shape

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    @property
    def mean(self) -> Tensor:
        result = (
            self.mu.exp() * self.sigma * np.pi / (self.sigma * np.pi).sin()
        )
        # Expectation diverges if sigma > 1
        return nd.where(
            self.sigma > 1.0, nd.full(result.shape, np.inf), result
        )

    def log_intensity(self, x: Tensor) -> Tensor:
        r"""
        Logarithm of the intensity (a.k.a. hazard) function.

        The intensity is defined as :math:`\lambda(x) = p(x) / S(x)`.

        We define :math:`z = (\log(x) - \mu) / \sigma` and obtain the intensity
        as :math:`\lambda(x) = sigmoid(z) / (\sigma * \log(x))`, or equivalently
        :math:`\log \lambda(x) = z - \log(1 + \exp(z)) - \log(\sigma) - \log(x)`.
        """
        log_x = x.clip(1e-20, np.inf).log()
        z = (log_x - self.mu) / self.sigma
        F = getF(x)
        return z - self.sigma.log() - F.Activation(z, "softrelu") - log_x

    def log_survival(self, x: Tensor) -> Tensor:
        r"""
        Logarithm of the survival function :math:`\log S(x) = \log(1 - CDF(x))`.

        We define :math:`z = (\log(x) - \mu) / \sigma` and obtain the survival
        function as :math:`S(x) = sigmoid(-z)`, or equivalently
        :math:`\log S(x) = -\log(1 + \exp(z))`.
        """
        log_x = x.clip(1e-20, np.inf).log()
        z = (log_x - self.mu) / self.sigma
        F = getF(x)
        return -F.Activation(z, "softrelu")

    def log_prob(self, x: Tensor) -> Tensor:
        return self.log_intensity(x) + self.log_survival(x)

    def sample(
        self,
        num_samples=None,
        dtype=np.float32,
        lower_bound: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Draw samples from the distribution.

        We generate samples as :math:`u \sim Uniform(0, 1), x = S^{-1}(u)`,
        where :math:`S^{-1}` is the inverse of the survival function
        :math:`S(x) = 1 - CDF(x)`.

        Parameters
        ----------
        num_samples
            Number of samples to generate.
        dtype
            Data type of the generated samples.
        lower_bound
            If None, generate samples as usual. If lower_bound is provided,
            all generated samples will be larger than the specified values.
            That is, we sample from `p(x | x > lower_bound)`.
            Shape: `(*batch_size)`

        Returns
        -------
        x
            Sampled inter-event times.
            Shape: `(num_samples, *batch_size)`
        """
        F = getF(self.mu)
        if num_samples is not None:
            sample_shape = (num_samples,) + self.batch_shape
        else:
            sample_shape = self.batch_shape
        u = F.uniform(0, 1, shape=sample_shape)
        # Make sure that the generated samples are larger than condition_above.
        # This is easy to ensure when using inverse-survival sampling: we simply
        # multiply `u ~ Uniform(0, 1)` by `S(y)` to ensure that `x > y`.
        with autograd.pause():
            if lower_bound is not None:
                survival = self.log_survival(lower_bound).exp()
                u = u * survival
            x = (self.mu + self.sigma * (F.log1p(-u) - F.log(u))).exp()
        return x


class LoglogisticOutput(TPPDistributionOutput):
    args_dim: Dict[str, int] = {"mu": 1, "sigma": 1}
    distr_cls: type = Loglogistic

    @classmethod
    def domain_map(cls, F, mu, sigma):
        r"""
        Maps raw tensors to valid arguments for constructing a log-logistic
        distribution.

        Parameters
        ----------
        F
            MXNet backend.
        mu
            Mean of the underlying logistic distribution.
            Shape `(*batch_shape, 1)`
        sigma
            Scale of the underlying logistic distribution.
            Shape `(*batch_shape, 1)`

        Returns
        -------
        Tuple[Tensor, Tensor]
            Two squeezed tensors of shape `(*batch_shape)`. The sigma parameter
            is strictly positive.
        """
        sigma = F.Activation(sigma, "softrelu")
        return mu.squeeze(axis=-1), sigma.squeeze(axis=-1)

    @property
    def event_shape(self) -> Tuple:
        return ()
