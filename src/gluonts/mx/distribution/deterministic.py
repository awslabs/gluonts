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

from typing import Dict, List, Optional, Tuple

import mxnet as mx
import numpy as np

from gluonts.core.component import DType, validated
from gluonts.mx import Tensor

from .distribution import Distribution, _sample_multiple, getF
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

    @property
    def F(self):
        return getF(self.value)

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
        self, num_samples: Optional[int] = None, dtype=np.float32
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


class DeterministicArgProj(mx.gluon.HybridBlock):
    def __init__(
        self,
        value: float,
        args_dim: Dict[str, int],
        dtype: DType = np.float32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.value = value
        self.args_dim = args_dim
        self.dtype = dtype

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, x: Tensor) -> Tuple[Tensor]:
        return (self.value * F.ones_like(x.sum(axis=-1)),)


class DeterministicOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"value": 1}
    distr_cls: type = Deterministic

    @validated()
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def get_args_proj(
        self, prefix: Optional[str] = None
    ) -> mx.gluon.HybridBlock:
        return DeterministicArgProj(
            value=self.value, args_dim=self.args_dim, dtype=self.dtype
        )

    @property
    def event_shape(self) -> Tuple:
        return ()
