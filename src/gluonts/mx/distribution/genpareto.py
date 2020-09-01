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
import math
from functools import partial
from typing import Dict, List, Optional, Tuple

# Third-party imports
import numpy as np

from gluonts.core.component import validated

from gluonts.model.common import Tensor
from gluonts.mx.distribution import Distribution
from gluonts.mx.distribution.distribution import (
    getF,
    softplus,
    _sample_multiple,
)
from gluonts.mx.distribution import uniform, box_cox_transform
from gluonts.mx.distribution.distribution_output import DistributionOutput


class GenPareto(Distribution):
    r"""
    Generalised Pareto distribution.

    Parameters
    ----------
    xi
        Tensor containing the xi shape parameters, of shape `(*batch_shape, *event_shape)`.
    beta
        Tensor containing the beta scale parameters, of shape `(*batch_shape, *event_shape)`.
    loc
        Tensor containing the (for-now positive) location parameters, of shape `(*batch_shape, *event_shape)`.
    F
    """

    is_reparameterizable = False

    @validated()
    def __init__(self, xi: Tensor, beta: Tensor, loc: Tensor) -> None:
        self.xi = xi
        self.beta = beta
        self.loc = loc

    @property
    def F(self):
        return getF(self.xi)

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
        xi, beta, loc = self.xi, self.beta, self.loc

        x_shifted = F.broadcast_div(x - loc, beta)
        return F.where(
            x < loc,
            -np.inf * F.ones_like(x),
            -(1 + F.reciprocal(xi)) * F.log1p(xi * x_shifted) - F.log(beta),
        )

    @property
    def mean(self) -> Tensor:
        F = self.F
        return F.where(
            self.xi < 1,
            self.loc + F.broadcast_div(self.beta, 1 - self.xi),
            np.nan * F.ones_like(self.xi),
        )

    @property
    def variance(self) -> Tensor:
        F = self.F
        xi, beta, loc = self.xi, self.beta, self.loc
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
        F = getF(self.xi)
        U = uniform.Uniform(F.array([0]), F.array([1])).sample(num_samples)
        boxcox = box_cox_transform.BoxCoxTransform(-self.xi, F.array([0]))
        samples = F.broadcast_add(
            F.broadcast_mul(-1 * boxcox.f(1 - U), self.beta), self.loc
        )
        return samples

    @property
    def args(self) -> List:
        return [self.xi, self.beta, self.loc]


class GenParetoOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"xi": 1, "beta": 1, "loc": 0}
    distr_cls: type = GenPareto

    @classmethod
    def domain_map(cls, F, xi, beta, loc):
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
        loc:
            Tensor of shape `(*batch_shape, 1)`

        Returns
        -------
        Tuple[Tensor, Tensor]:
            Two squeezed tensors, of shape `(*batch_shape)`: both have entries mapped to the
            positive orthant.
        """
        epsilon = np.finfo(cls._dtype).eps
        beta = softplus(F, beta) + epsilon
        return xi.squeeze(axis=-1), beta.squeeze(axis=-1), loc.squeeze(axis=1)

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def value_in_support(self) -> float:
        return 0.5
