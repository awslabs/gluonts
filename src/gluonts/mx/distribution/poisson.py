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
from typing import Dict, Optional, Tuple, List

# Third-party imports
import numpy as np

# First-party imports
from gluonts.model.common import Tensor
from gluonts.core.component import validated

# Relative imports
from .distribution import Distribution, _sample_multiple, getF, softplus
from .distribution_output import DistributionOutput


class Poisson(Distribution):
    r"""
    Poisson distribution, i.e. the distribution of the number of
    successes in a specified region.

    Parameters
    ----------
    rate
        Tensor containing the means, of shape `(*batch_shape, *event_shape)`.
    F
    """

    is_reparameterizable = False

    @validated()
    def __init__(self, rate: Tensor, F=None) -> None:
        self.rate = rate
        self.F = F if F else getF(rate)

    @property
    def batch_shape(self) -> Tuple:
        return self.rate.shape

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    def log_prob(self, x: Tensor) -> Tensor:
        F = self.F
        ll = x * F.log(self.rate) - F.gammaln(x + 1.0) - self.rate
        return ll

    @property
    def mean(self) -> Tensor:
        return self.rate

    @property
    def stddev(self) -> Tensor:
        return self.F.sqrt(self.rate)

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        def s(rate: Tensor) -> Tensor:
            return self.F.random.poisson(lam=rate, dtype=dtype)

        return _sample_multiple(s, rate=self.rate, num_samples=num_samples)

    @property
    def args(self) -> List:
        return [self.rate]


class PoissonOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"rate": 1}
    distr_cls: type = Poisson

    @classmethod
    def domain_map(cls, F, rate):
        rate = softplus(F, rate) + 1e-8
        return rate.squeeze(axis=-1)

    # Overwrites the parent class method.
    # We cannot scale using the affine transformation since Poisson should return integers.
    # Instead we scale the parameters.
    def distribution(
        self,
        distr_args,
        loc: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ) -> Poisson:
        rate = distr_args
        if scale is None:
            return Poisson(rate)
        else:
            F = getF(rate)
            rate = F.broadcast_mul(rate, scale)
            return Poisson(rate, F)

    @property
    def event_shape(self) -> Tuple:
        return ()
