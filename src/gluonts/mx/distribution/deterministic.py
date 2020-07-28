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

# First-party imports
from gluonts.model.common import Tensor
from gluonts.core.component import validated
from gluonts.support.util import erf, erfinv

# Relative imports
from .distribution import Distribution, _sample_multiple, getF, softplus
from .distribution_output import DistributionOutput


class Deterministic(Distribution):
    r"""
    Deterministic/Degenerate distribution.

    Parameters
    ----------
    value
        Tensor containing the values, of shape `(*batch_shape, *event_shape)`.
    F
    """

    is_reparameterizable = True

    @validated()
    def __init__(self, value: Tensor) -> None:
        self.value = value
        self.F = getF(value)

    @property
    def batch_shape(self) -> Tuple:
        return self.value.shape

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    def log_prob(self, x: Tensor) -> Tensor:
        F = self.F
        value = self.value
        is_both_nan = F.broadcast_logical_and(x != x, value != value)
        is_equal_or_both_nan = F.broadcast_logical_or(
            (x == value), is_both_nan
        )
        return F.log(is_equal_or_both_nan)

    @property
    def mean(self) -> Tensor:
        return self.value

    @property
    def stddev(self) -> Tensor:
        return self.value.zeros_like()

    def cdf(self, x):
        F = self.F
        value = self.value
        is_both_nan = F.broadcast_logical_and(
            F.contrib.isnan(x), F.contrib.isnan(value)
        )
        is_greater_equal_or_both_nan = F.broadcast_logical_or(
            (x >= value), is_both_nan
        )
        return is_greater_equal_or_both_nan

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.int32
    ) -> Tensor:
        return _sample_multiple(
            lambda value: value, value=self.value, num_samples=num_samples
        ).astype(dtype=dtype)

    def quantile(self, level: Tensor) -> Tensor:
        F = self.F
        # we consider level to be an independent axis and so expand it
        # to shape (num_levels, 1, 1, ...)

        for _ in range(self.all_dim):
            level = level.expand_dims(axis=-1)

        quantiles = F.broadcast_mul(self.value, level.ones_like())
        level = F.broadcast_mul(quantiles.ones_like(), level)

        minus_inf = -quantiles.ones_like() / 0.0
        quantiles = F.where(
            F.broadcast_logical_or(level != 0, F.contrib.isnan(quantiles)),
            quantiles,
            minus_inf,
        )

        nans = level.zeros_like() / 0.0
        quantiles = F.where(level != level, nans, quantiles)

        return quantiles

    @property
    def args(self) -> List:
        return [self.value]


class DeterministicOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"value": 1}
    distr_cls: type = Deterministic

    @classmethod
    def domain_map(cls, F, value):
        r"""
        Maps raw tensors to valid arguments for constructing a Gaussian
        distribution.

        Parameters
        ----------
        F
        value
            Tensor of shape `(*batch_shape, 1)`

        Returns
        -------
        Tuple[Tensor, Tensor]
            Two squeezed tensors, of shape `(*batch_shape)`: the first has the
            same entries as `mu` and the second has entries mapped to the
            positive orthant.
        """
        return value.squeeze(axis=-1)

    @property
    def event_shape(self) -> Tuple:
        return ()
