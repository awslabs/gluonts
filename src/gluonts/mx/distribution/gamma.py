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

from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np

from gluonts.core.component import validated
from gluonts.mx import Tensor

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
    def __init__(self, alpha: Tensor, beta: Tensor) -> None:
        self.alpha = alpha
        self.beta = beta

    @property
    def F(self):
        return getF(self.alpha)

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

        def gamma_log_prob(x, alpha, beta):
            return (
                alpha * F.log(beta)
                - F.gammaln(alpha)
                + (alpha - 1) * F.log(x)
                - beta * x
            )

        """
        The gamma_log_prob(x) above returns NaNs for x<=0. Wherever there are NaN in either of the F.where() conditional
        vectors, then F.where() returns NaN at that entry as well, due to its indicator function multiplication: 
        1*f(x) + np.nan*0 = nan, since np.nan*0 return nan. 
        Therefore replacing gamma_log_prob(x) with gamma_log_prob(abs(x) mitigates nan returns in cases of x<=0 without 
        altering the value in cases of x>0. 
        This is a known issue in pytorch as well https://github.com/pytorch/pytorch/issues/12986.
        """
        # mask zeros to prevent NaN gradients for x==0
        x_masked = F.where(x == 0, x.ones_like() * 0.5, x)

        return F.where(
            x > 0,
            gamma_log_prob(F.abs(x_masked), alpha, beta),
            -(10.0 ** 15) * F.ones_like(x),
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
        F = self.F

        samples = _sample_multiple(
            partial(F.sample_gamma, dtype=dtype),
            alpha=self.alpha,
            beta=1.0 / self.beta,
            num_samples=num_samples,
        )
        return F.clip(data=samples, a_min=epsilon, a_max=np.finfo(dtype).max)

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
