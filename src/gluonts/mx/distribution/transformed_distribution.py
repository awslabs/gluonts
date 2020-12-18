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

from typing import Any, List, Optional, Tuple

import mxnet as mx
import numpy as np
from mxnet import autograd

from gluonts.core.component import validated
from gluonts.mx import Tensor

from . import bijection as bij
from .distribution import Distribution, _index_tensor, getF
from .bijection import AffineTransformation


class TransformedDistribution(Distribution):
    r"""
    A distribution obtained by applying a sequence of transformations on top
    of a base distribution.
    """

    @validated()
    def __init__(
        self, base_distribution: Distribution, transforms: List[bij.Bijection]
    ) -> None:
        self.base_distribution = base_distribution
        self.transforms = transforms
        self.is_reparameterizable = self.base_distribution.is_reparameterizable

        # use these to cache shapes and avoid recomputing all steps
        # the reason we cannot do the computations here directly
        # is that this constructor would fail in mx.symbol mode
        self._event_dim: Optional[int] = None
        self._event_shape: Optional[Tuple] = None
        self._batch_shape: Optional[Tuple] = None

    @property
    def F(self):
        return self.base_distribution.F

    @property
    def support_min_max(self) -> Tuple[Tensor, Tensor]:
        F = self.F
        lb, ub = self.base_distribution.support_min_max
        for t in self.transforms:
            _lb = t.f(lb)
            _ub = t.f(ub)
            lb = F.minimum(_lb, _ub)
            ub = F.maximum(_lb, _ub)
        return lb, ub

    def _slice_bijection(
        self, trans: bij.Bijection, item: Any
    ) -> bij.Bijection:
        from .box_cox_transform import BoxCoxTransform

        if isinstance(trans, bij.AffineTransformation):
            loc = (
                _index_tensor(trans.loc, item)
                if trans.loc is not None
                else None
            )
            scale = (
                _index_tensor(trans.scale, item)
                if trans.scale is not None
                else None
            )
            return bij.AffineTransformation(loc=loc, scale=scale)
        elif isinstance(trans, BoxCoxTransform):
            return BoxCoxTransform(
                _index_tensor(trans.lambda_1, item),
                _index_tensor(trans.lambda_2, item),
            )
        elif isinstance(trans, bij.InverseBijection):
            return bij.InverseBijection(
                self._slice_bijection(trans._bijection, item)
            )
        else:
            return trans

    def __getitem__(self, item):
        bd_slice = self.base_distribution[item]
        trans_slice = [self._slice_bijection(t, item) for t in self.transforms]
        return TransformedDistribution(bd_slice, trans_slice)

    @property
    def event_dim(self):
        if self._event_dim is None:
            self._event_dim = max(
                [self.base_distribution.event_dim]
                + [t.event_dim for t in self.transforms]
            )
        assert isinstance(self._event_dim, int)
        return self._event_dim

    @property
    def batch_shape(self) -> Tuple:
        if self._batch_shape is None:
            shape = (
                self.base_distribution.batch_shape
                + self.base_distribution.event_shape
            )
            self._batch_shape = shape[: len(shape) - self.event_dim]
        assert isinstance(self._batch_shape, tuple)
        return self._batch_shape

    @property
    def event_shape(self) -> Tuple:
        if self._event_shape is None:
            shape = (
                self.base_distribution.batch_shape
                + self.base_distribution.event_shape
            )
            self._event_shape = shape[len(shape) - self.event_dim :]
        assert isinstance(self._event_shape, tuple)
        return self._event_shape

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        with autograd.pause():
            s = self.base_distribution.sample(
                num_samples=num_samples, dtype=dtype
            )
            for t in self.transforms:
                s = t.f(s)
            return s

    def sample_rep(
        self, num_samples: Optional[int] = None, dtype=np.float
    ) -> Tensor:
        s = self.base_distribution.sample_rep(
            num_samples=num_samples, dtype=dtype
        )
        for t in self.transforms:
            s = t.f(s)
        return s

    def log_prob(self, y: Tensor) -> Tensor:
        F = getF(y)
        lp = 0.0
        x = y
        for t in self.transforms[::-1]:
            x = t.f_inv(y)
            ladj = t.log_abs_det_jac(x, y)
            lp -= sum_trailing_axes(F, ladj, self.event_dim - t.event_dim)
            y = x

        return self.base_distribution.log_prob(x) + lp

    def cdf(self, y: Tensor) -> Tensor:
        x = y
        sign = 1.0
        for t in self.transforms[::-1]:
            x = t.f_inv(x)
            sign = sign * t.sign
        f = self.base_distribution.cdf(x)
        return sign * (f - 0.5) + 0.5

    def quantile(self, level: Tensor) -> Tensor:
        F = getF(level)

        sign = 1.0
        for t in self.transforms:
            sign = sign * t.sign

        if not isinstance(sign, (mx.nd.NDArray, mx.sym.Symbol)):
            level = level if sign > 0 else (1.0 - level)
            q = self.base_distribution.quantile(level)
        else:
            # level.shape = (#levels,)
            # q_pos.shape = (#levels, batch_size, ...)
            # sign.shape = (batch_size, ...)
            q_pos = self.base_distribution.quantile(level)
            q_neg = self.base_distribution.quantile(1.0 - level)
            cond = F.broadcast_greater(sign, sign.zeros_like())
            cond = F.broadcast_add(cond, q_pos.zeros_like())
            q = F.where(cond, q_pos, q_neg)

        for t in self.transforms:
            q = t.f(q)
        return q


class AffineTransformedDistribution(TransformedDistribution):
    @validated()
    def __init__(
        self,
        base_distribution: Distribution,
        loc: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ) -> None:
        super().__init__(base_distribution, [AffineTransformation(loc, scale)])

        self.loc = loc
        self.scale = scale

    @property
    def mean(self) -> Tensor:
        return (
            self.base_distribution.mean
            if self.loc is None
            else self.base_distribution.mean + self.loc
        )

    @property
    def stddev(self) -> Tensor:
        return (
            self.base_distribution.stddev
            if self.scale is None
            else self.base_distribution.stddev * self.scale
        )

    @property
    def variance(self) -> Tensor:
        # TODO: cover the multivariate case here too
        return (
            self.base_distribution.variance
            if self.scale is None
            else self.base_distribution.variance * self.scale ** 2
        )

    # TODO: crps


def sum_trailing_axes(F, x: Tensor, k: int) -> Tensor:
    for _ in range(k):
        x = F.sum(x, axis=-1)
    return x
