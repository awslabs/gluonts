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
from gluonts.mx.distribution import Distribution, box_cox_transform, uniform
from gluonts.mx.distribution.distribution import (
    MAX_SUPPORT_VAL,
    _sample_multiple,
    getF,
    softplus,
)
from gluonts.mx.distribution.distribution_output import DistributionOutput

from .distribution import Distribution


class GenPareto(Distribution):
    r"""
    Generalised Pareto distribution.

    Parameters
    ----------
    xi
        Tensor containing the xi shape parameters, of shape `(*batch_shape, *event_shape)`.
    beta
        Tensor containing the beta scale parameters, of shape `(*batch_shape, *event_shape)`.
    """

    is_reparameterizable = False

    @validated()
    def __init__(self, xi: Tensor, beta: Tensor) -> None:
        self.xi = xi
        self.beta = beta

    @property
    def F(self):
        return getF(self.xi)

    @property
    def support_min_max(self) -> Tuple[Tensor, Tensor]:
        F = self.F
        return (
            F.zeros(self.batch_shape),
            F.ones(self.batch_shape) * MAX_SUPPORT_VAL,
        )

    @property
    def batch_shape(self) -> Tuple:
        return self.xi.shape

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    def log_prob(self, x: Tensor) -> Tensor:
        F = self.F
        xi, beta = self.xi, self.beta

        def genpareto_log_prob(x, xi, beta):
            x_shifted = F.broadcast_div(x, beta)
            return -(1 + F.reciprocal(xi)) * F.log1p(xi * x_shifted) - F.log(
                beta
            )

        """
        The genpareto_log_prob(x) above returns NaNs for x<0. Wherever there are NaN in either of the F.where() conditional
        vectors, then F.where() returns NaN at that entry as well, due to its indicator function multiplication: 
        1*f(x) + np.nan*0 = nan, since np.nan*0 return nan. 
        Therefore replacing genpareto_log_prob(x) with genpareto_log_prob(abs(x) mitigates nan returns in cases of x<0 without 
        altering the value in cases of x>=0. 
        This is a known issue in pytorch as well https://github.com/pytorch/pytorch/issues/12986.
        """
        return F.where(
            x < 0,
            -(10.0 ** 15) * F.ones_like(x),
            genpareto_log_prob(F.abs(x), xi, beta),
        )

    def cdf(self, x: Tensor) -> Tensor:
        F = self.F
        x_shifted = F.broadcast_div(x, self.beta)
        u = 1 - F.power(1 + self.xi * x_shifted, -F.reciprocal(self.xi))
        return u

    def quantile(self, level: Tensor):
        F = self.F
        # we consider level to be an independent axis and so expand it
        # to shape (num_levels, 1, 1, ...)
        for _ in range(self.all_dim):
            level = level.expand_dims(axis=-1)

        x_shifted = F.broadcast_div(F.power(1 - level, -self.xi) - 1, self.xi)
        x = F.broadcast_mul(x_shifted, self.beta)
        return x

    @property
    def mean(self) -> Tensor:
        F = self.F
        return F.where(
            self.xi < 1,
            F.broadcast_div(self.beta, 1 - self.xi),
            np.nan * F.ones_like(self.xi),
        )

    @property
    def variance(self) -> Tensor:
        F = self.F
        xi, beta = self.xi, self.beta
        return F.where(
            xi < 1 / 2,
            F.broadcast_div(
                beta ** 2, F.broadcast_mul((1 - xi) ** 2, (1 - 2 * xi))
            ),
            np.nan * F.ones_like(xi),
        )

    @property
    def stddev(self) -> Tensor:
        return self.F.sqrt(self.variance)

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        def s(xi: Tensor, beta: Tensor) -> Tensor:
            F = getF(xi)
            sample_U = uniform.Uniform(
                F.zeros_like(xi), F.ones_like(xi)
            ).sample()
            boxcox = box_cox_transform.BoxCoxTransform(-xi, F.array([0]))
            sample_X = -1 * boxcox.f(1 - sample_U) * beta
            return sample_X

        samples = _sample_multiple(
            s,
            xi=self.xi,
            beta=self.beta,
            num_samples=num_samples,
        )
        return self.F.clip(
            data=samples, a_min=np.finfo(dtype).eps, a_max=np.finfo(dtype).max
        )

    @property
    def args(self) -> List:
        return [self.xi, self.beta]


class GenParetoOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"xi": 1, "beta": 1}
    distr_cls: type = GenPareto

    @classmethod
    def domain_map(cls, F, xi, beta):
        r"""
        Maps raw tensors to valid arguments for constructing a Generalized Pareto
        distribution.

        Parameters
        ----------
        F:
        xi:
            Tensor of shape `(*batch_shape, 1)`
        beta:
            Tensor of shape `(*batch_shape, 1)`

        Returns
        -------
        Tuple[Tensor, Tensor]:
            Two squeezed tensors, of shape `(*batch_shape)`: both have entries mapped to the
            positive orthant.
        """
        epsilon = np.finfo(cls._dtype).eps
        xi = softplus(F, xi) + epsilon
        beta = softplus(F, beta) + epsilon
        return xi.squeeze(axis=-1), beta.squeeze(axis=-1)

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def value_in_support(self) -> float:
        return 0.5
