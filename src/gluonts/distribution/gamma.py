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
from typing import Dict, Optional, Tuple, List

# Third-party imports
import numpy as np

# First-party imports
from gluonts.model.common import Tensor
from gluonts.support.util import erf, erfinv
from gluonts.core.component import validated

# Relative imports
from .distribution import Distribution, _sample_multiple, getF, softplus
from .distribution_output import DistributionOutput


class Gamma(Distribution):
    r"""
    Gamma distribution.

    Parameters
    ----------
    alpha
        Tensor containing the shape parameters, of shape `(*batch_shape, *event_shape)`.
    beta
        Tensor containing the rate parameters, of shape `(*batch_shape, *event_shape)`.
    F
    """

    is_reparameterizable = False

    @validated()
    def __init__(self, alpha: Tensor, beta: Tensor, F=None) -> None:
        self.alpha = alpha
        self.beta = beta
        self.F = (
            F if F else getF(alpha)
        )  # assuming alpha and beta of same type

    @property
    def batch_shape(self) -> Tuple:
        return self.alpha.shape

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    def log_prob(self, x: Tensor) -> Tensor:
        F = self.F
        alpha, beta = self.alpha, self.beta

        return (
            alpha * F.log(beta)
            - F.gammaln(alpha)
            + (alpha - 1) * F.log(x)
            - beta * x
        )

    @property
    def mean(self) -> Tensor:
        return self.alpha / self.beta

    @property
    def stddev(self) -> Tensor:
        return self.F.sqrt(self.alpha) / self.beta

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        epsilon = np.finfo(dtype).eps  # machine epsilon

        samples = _sample_multiple(
            partial(self.F.sample_gamma, dtype=dtype),
            alpha=self.alpha,
            beta=1.0 / self.beta,
            num_samples=num_samples,
        )
        return self.F.clip(
            data=samples, a_min=epsilon, a_max=np.finfo(dtype).max
        )

    @property
    def args(self) -> List:
        return [self.alpha, self.beta]


class GammaOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"alpha": 1, "beta": 1}
    distr_cls: type = Gamma

    @classmethod
    def domain_map(cls, F, alpha, beta):
        r"""
        Maps raw tensors to valid arguments for constructing a Gamma
        distribution.

        Parameters
        ----------
        F
        alpha
            Tensor of shape `(*batch_shape, 1)`
        beta
            Tensor of shape `(*batch_shape, 1)`

        Returns
        -------
        Tuple[Tensor, Tensor]
            Two squeezed tensors, of shape `(*batch_shape)`: both have entries mapped to the
            positive orthant.
        """
        epsilon = np.finfo(cls._dtype).eps  # machine epsilon

        alpha = softplus(F, alpha) + epsilon
        beta = softplus(F, beta) + epsilon
        return alpha.squeeze(axis=-1), beta.squeeze(axis=-1)

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def value_in_support(self) -> float:
        return 0.5
