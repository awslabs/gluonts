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
from typing import Dict, Optional, Tuple

# Third-party imports
import numpy as np
from mxnet import autograd, nd

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor
from gluonts.mx.distribution.distribution import getF

# Relative imports
from .base import TPPDistribution, TPPDistributionOutput


class Weibull(TPPDistribution):
    """
    Weibull distribution.

    We use the parametrization of the Weibull distribution using the rate
    parameter `b > 0` and the shape parameter `k > 0`. The PDF is
    `p(x) = b * k * x^(k - 1) * exp(-b * x^(k))`. An alternative parametrization
    is often used (e.g. on Wikipedia), where we use the scale parameter
    `lambda > 0` and the shape parameter `k > 0`, and `lambda = b^(-1/k)`.
    """

    is_reparametrizable = True

    @validated()
    def __init__(self, rate: Tensor, shape: Tensor) -> None:
        self.rate = rate
        self.shape = shape

    @property
    def batch_shape(self) -> Tuple:
        return self.rate.shape

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    @property
    def mean(self) -> Tensor:
        return nd.power(self.rate, -1.0 / self.shape) * nd.gamma(
            1.0 + 1.0 / self.shape
        )

    def log_intensity(self, x: Tensor) -> Tensor:
        """
        Logarithm of the intensity (a.k.a. hazard) function.

        The intensity is defined as `lambda(x) = p(x) / S(x)`.

        The intensity of the Weibull distribution is
        `lambda(x) = b * k * x^(k - 1)`.
        """
        log_x = x.clip(1e-10, np.inf).log()
        return self.rate.log() + self.shape.log() + (self.shape - 1) * log_x

    def log_survival(self, x: Tensor) -> Tensor:
        """
        Logarithm of the survival function `log(S(x)) = log(1 - CDF(x))`.

        The survival function of the Weibull distribution is
        `S(x) = exp(-b * x^k)`.
        """
        # We need to add eps=1e-10 to avoid numerical instability of pow()
        return -self.rate * (x + 1e-10) ** self.shape

    def log_prob(self, x: Tensor) -> Tensor:
        return self.log_intensity(x) + self.log_survival(x)

    def sample(
        self,
        num_samples=None,
        dtype=np.float32,
        lower_bound: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Draw samples from the distribution.

        We generate samples as `u ~ Uniform(0, 1), x = S^{-1}(u)`,
        where `S^{-1}` is the inverse of the survival function
        `S(x) = 1 - CDF(x)`.

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
        F = getF(self.rate)
        if num_samples is not None:
            sample_shape = (num_samples,) + self.batch_shape
        else:
            sample_shape = self.batch_shape
        u = F.uniform(0, 1, shape=sample_shape)
        # Make sure that the generated samples are larger than condition_above.
        # This is easy to ensure when using inverse-survival sampling: we simply
        # multiply `u ~ Unif(0, 1)` by `S(y)` to ensure that `x > y`.
        with autograd.pause():
            if lower_bound is not None:
                survival = self.log_survival(lower_bound).exp()
                u = u * survival
            x = (-u.log() / self.rate) ** (1.0 / self.shape)
        return x


class WeibullOutput(TPPDistributionOutput):
    args_dim: Dict[str, int] = {"rate": 1, "shape": 1}
    distr_cls: type = Weibull

    @classmethod
    def domain_map(cls, F, rate, shape):
        r"""
        Maps raw tensors to valid arguments for constructing a Weibull
        distribution.

        Parameters
        ----------
        F
            MXNet backend.
        rate
            Rate (inverse scale) parameter of the Weibull distribution.
            Shape `(*batch_shape, 1)`
        shape
            Shape parameter of the Weibull distribution.
            Shape `(*batch_shape, 1)`

        Returns
        -------
        Tuple[Tensor, Tensor]
            Two squeezed tensors of shape `(*batch_shape)`. Both tensors are
            strictly positive.
        """
        rate = F.Activation(rate, "softrelu")
        shape = F.Activation(shape, "softrelu")
        return rate.squeeze(axis=-1), shape.squeeze(axis=-1)

    @property
    def event_shape(self) -> Tuple:
        return ()
