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


class Uniform(Distribution):
    r"""
    Uniform distribution.

    Parameters
    ----------
    low
        Tensor containing the lower bound of the distribution domain.
    high
        Tensor containing the higher bound of the distribution domain.
    F
    """

    is_reparameterizable = True

    @validated()
    def __init__(self, low: Tensor, high: Tensor) -> None:
        self.low = low
        self.high = high

    @property
    def F(self):
        return getF(self.low)

    @property
    def batch_shape(self) -> Tuple:
        return self.low.shape

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    def log_prob(self, x: Tensor) -> Tensor:
        F = self.F
        is_in_range = F.broadcast_greater_equal(
            x, self.low
        ) * F.broadcast_lesser(x, self.high)
        return F.log(is_in_range) - F.log(self.high - self.low)

    @property
    def mean(self) -> Tensor:
        return (self.high + self.low) / 2

    @property
    def stddev(self) -> Tensor:
        return (self.high - self.low) / (12 ** 0.5)

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        return _sample_multiple(
            partial(self.F.sample_uniform, dtype=dtype),
            low=self.low,
            high=self.high,
            num_samples=num_samples,
        )

    def sample_rep(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        def s(low: Tensor, high: Tensor) -> Tensor:
            raw_samples = self.F.sample_uniform(
                low=low.zeros_like(), high=high.ones_like(), dtype=dtype
            )
            return low + raw_samples * (high - low)

        return _sample_multiple(
            s, low=self.low, high=self.high, num_samples=num_samples
        )

    def cdf(self, x: Tensor) -> Tensor:
        return self.F.broadcast_div(x - self.low, self.high - self.low)

    def quantile(self, level: Tensor) -> Tensor:
        F = self.F
        for _ in range(self.all_dim):
            level = level.expand_dims(axis=-1)
        return F.broadcast_add(
            F.broadcast_mul(self.high - self.low, level), self.low
        )

    @property
    def args(self) -> List:
        return [self.low, self.high]


class UniformOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"low": 1, "width": 1}
    distr_cls: type = Uniform

    @classmethod
    def domain_map(cls, F, low, width):
        high = low + softplus(F, width)
        return low.squeeze(axis=-1), high.squeeze(axis=-1)

    @property
    def event_shape(self) -> Tuple:
        return ()
