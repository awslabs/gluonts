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
from typing import Dict, Optional, Tuple, List

# Third-party imports
import numpy as np

# First-party imports
from gluonts.model.common import Tensor

from gluonts.core.component import validated

# Relative imports
from .distribution import (
    Distribution,
    _sample_multiple,
    getF,
    nans_like,
    softplus,
)
from .distribution_output import DistributionOutput


class PeakOverThresholdGeneralizedPareto(Distribution):
    r"""
    The Generalized Pareto is a continuous distribution defined on the real line.
    This distribution is often used to model the tails of other distributions.


    Parameters
    ----------
      scale: The scale of the distribution. GeneralizedPareto is a
        location-scale distribution, so doubling the `scale` doubles a sample
        and halves the density. Strictly positive floating point `Tensor`. Must
        broadcast with `concentration`.
        
      concentration: The shape parameter of the distribution,
      mapped to `concentration > 0` for heavy tails.
    F
    """

    is_reparameterizable = False

    @validated()
    def __init__(self, scale: Tensor, concentration: Tensor, F=None) -> None:
        self.scale = scale
        self.concentration = concentration
        self.F = F if F else getF(concentration)

    @property
    def batch_shape(self) -> Tuple:
        return self.concentration.shape

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    def log_prob(self, x: Tensor) -> Tensor:
        F = self.F
        z = x / self.scale

        ll = -F.log(self.scale) - (
            (self.concentration + 1) / self.concentration
        ) * F.log1p(self.concentration * z)

        return ll

    @property
    def mean(self) -> Tensor:
        # Mean is only defined for concentration < 1. How to handle that?
        mu = self.scale / (1 - self.concentration)
        return mu

    @property
    def stddev(self) -> Tensor:
        # Variance is only defined for concentration < 1/2. How?
        F = self.F
        return self.scale / (
            (1 - self.concentration) * self.F.sqrt(1 - 2 * self.concentration)
        )

    def cdf(self, x: Tensor) -> Tensor:
        F = self.F
        z = x / self.scale
        u = 1 - F.power(1 + self.concentration * z, -1 / self.concentration)
        return u

    def base_distribution_quantile(
        self,
        level: Tensor,
        threshold: Tensor,
        peak_ratio: Tensor,
        transforms,
        below=False,
    ) -> Tensor:
        """
        Computes the tail quantile of the base distribution using the fitted extreme value distribution. 
        """

        F = self.F
        sgn = -1 if below else 1

        # Base GPD quantile
        base_q = (self.scale / self.concentration) * (
            F.power(level / peak_ratio, -self.concentration) - 1
        )

        # Reversing the transforms
        for t in transforms:
            base_q = t.f(base_q)

        # Applying over/below threshold
        q = threshold + sgn * base_q

        return q

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        def s(scale: Tensor, concentration: Tensor) -> Tensor:
            F = self.F
            ones = concentration.ones_like()

            smpl = (
                scale
                * (
                    F.power(
                        F.random.uniform(0 * ones, 1 * ones, dtype=dtype),
                        -concentration,
                    )
                    - 1
                )
                / concentration
            )

            return smpl

        return _sample_multiple(
            s,
            scale=self.scale,
            concentration=self.concentration,
            num_samples=num_samples,
        )

    @property
    def args(self) -> List:
        return [self.scale, self.concentration]


class PeakOverThresholdGeneralizedParetoOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"scale": 1, "concentration": 1}
    distr_cls: type = PeakOverThresholdGeneralizedPareto

    @classmethod
    def domain_map(cls, F, scale, concentration):
        scale = 1e-4 + softplus(F, scale)
        concentration = 1e-4 + softplus(F, concentration)
        return scale.squeeze(axis=-1), concentration.squeeze(axis=-1)

    @property
    def event_shape(self) -> Tuple:
        return ()
