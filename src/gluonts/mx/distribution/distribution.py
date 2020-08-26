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
from typing import Any, List, Optional, Tuple

# Third-party imports
import mxnet as mx
import numpy as np
from mxnet import autograd

# First-party imports
from gluonts.model.common import Tensor


def nans_like(x: Tensor) -> Tensor:
    return x.zeros_like() / 0.0


def softplus(F, x: Tensor) -> Tensor:
    return F.Activation(x, act_type="softrelu")


def getF(var: Tensor):
    if isinstance(var, mx.nd.NDArray):
        return mx.nd
    elif isinstance(var, mx.sym.Symbol):
        return mx.sym
    else:
        raise RuntimeError("var must be instance of NDArray or Symbol in getF")


def _index_tensor(x: Tensor, item: Any) -> Tensor:
    """
    """
    squeeze: List[int] = []
    if not isinstance(item, tuple):
        item = (item,)

    saw_ellipsis = False

    for i, item_i in enumerate(item):
        axis = i - len(item) if saw_ellipsis else i
        if isinstance(item_i, int):
            if item_i != -1:
                x = x.slice_axis(axis=axis, begin=item_i, end=item_i + 1)
            else:
                x = x.slice_axis(axis=axis, begin=-1, end=None)
            squeeze.append(axis)
        elif item_i == slice(None):
            continue
        elif item_i == Ellipsis:
            saw_ellipsis = True
            continue
        elif isinstance(item_i, slice):
            assert item_i.step is None
            start = item_i.start if item_i.start is not None else 0
            x = x.slice_axis(axis=axis, begin=start, end=item_i.stop)
        else:
            raise RuntimeError(f"invalid indexing item: {item}")
    if len(squeeze):
        x = x.squeeze(axis=tuple(squeeze))
    return x


class Distribution:
    r"""
    A class representing probability distributions.
    """

    arg_names: Tuple
    is_reparameterizable = False

    def log_prob(self, x: Tensor) -> Tensor:
        r"""
        Compute the log-density of the distribution at `x`.

        Parameters
        ----------
        x
            Tensor of shape `(*batch_shape, *event_shape)`.

        Returns
        -------
        Tensor
            Tensor of shape `batch_shape` containing the log-density of the
            distribution for each event in `x`.
        """
        raise NotImplementedError()

    def crps(self, x: Tensor) -> Tensor:
        r"""
        Compute the *continuous rank probability score* (CRPS) of `x` according
        to the distribution.

        Parameters
        ----------
        x
            Tensor of shape `(*batch_shape, *event_shape)`.

        Returns
        -------
        Tensor
            Tensor of shape `batch_shape` containing the CRPS score,
            according to the distribution, for each event in `x`.
        """
        raise NotImplementedError()

    def loss(self, x: Tensor) -> Tensor:
        r"""
        Compute the loss at `x` according to the distribution.

        By default, this method returns the negative of `log_prob`. For some
        distributions, however, the log-density is not easily computable
        and therefore other loss functions are computed.

        Parameters
        ----------
        x
            Tensor of shape `(*batch_shape, *event_shape)`.

        Returns
        -------
        Tensor
            Tensor of shape `batch_shape` containing the value of the loss
            for each event in `x`.
        """
        return -self.log_prob(x)

    def prob(self, x: Tensor) -> Tensor:
        r"""
        Compute the density of the distribution at `x`.

        Parameters
        ----------
        x
            Tensor of shape `(*batch_shape, *event_shape)`.

        Returns
        -------
        Tensor
            Tensor of shape `batch_shape` containing the density of the
            distribution for each event in `x`.
        """
        return self.log_prob(x).exp()

    @property
    def batch_shape(self) -> Tuple:
        r"""
        Layout of the set of events contemplated by the distribution.

        Invoking `sample()` from a distribution yields a tensor of shape
        `batch_shape + event_shape`, and computing `log_prob` (or `loss`
        more in general) on such sample will yield a tensor of shape
        `batch_shape`.

        This property is available in general only in mx.ndarray mode,
        when the shape of the distribution arguments can be accessed.
        """
        raise NotImplementedError()

    @property
    def event_shape(self) -> Tuple:
        r"""
        Shape of each individual event contemplated by the distribution.

        For example, distributions over scalars have `event_shape = ()`,
        over vectors have `event_shape = (d, )` where `d` is the length
        of the vectors, over matrices have `event_shape = (d1, d2)`, and
        so on.

        Invoking `sample()` from a distribution yields a tensor of shape
        `batch_shape + event_shape`.

        This property is available in general only in mx.ndarray mode,
        when the shape of the distribution arguments can be accessed.
        """
        raise NotImplementedError()

    @property
    def event_dim(self) -> int:
        r"""
        Number of event dimensions, i.e., length of the `event_shape` tuple.

        This is `0` for distributions over scalars, `1` over vectors,
        `2` over matrices, and so on.
        """
        raise NotImplementedError()

    @property
    def batch_dim(self) -> int:
        r"""
        Number of batch dimensions, i.e., length of the `batch_shape` tuple.
        """
        return len(self.batch_shape)

    @property
    def all_dim(self) -> int:
        r"""
        Number of overall dimensions.
        """
        return self.batch_dim + self.event_dim

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        r"""
        Draw samples from the distribution.

        If num_samples is given the first dimension of the output will be
        num_samples.

        Parameters
        ----------
        num_samples
            Number of samples to to be drawn.
        dtype
            Data-type of the samples.

        Returns
        -------
        Tensor
            A tensor containing samples. This has shape
            `(*batch_shape, *eval_shape)` if `num_samples = None`
            and  `(num_samples, *batch_shape, *eval_shape)` otherwise.
        """
        with autograd.pause():
            var = self.sample_rep(num_samples=num_samples, dtype=dtype)
            F = getF(var)
            return F.BlockGrad(var)

    def sample_rep(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        raise NotImplementedError()

    @property
    def args(self) -> List:
        raise NotImplementedError()

    @property
    def mean(self) -> Tensor:
        r"""
        Tensor containing the mean of the distribution.
        """
        raise NotImplementedError()

    @property
    def stddev(self) -> Tensor:
        r"""
        Tensor containing the standard deviation of the distribution.
        """
        raise NotImplementedError()

    @property
    def variance(self) -> Tensor:
        r"""
        Tensor containing the variance of the distribution.
        """
        return self.stddev.square()

    def cdf(self, x: Tensor) -> Tensor:
        r"""
        Returns the value of the cumulative distribution function evaluated at x
        """
        raise NotImplementedError()

    def quantile(self, level: Tensor) -> Tensor:
        r"""

        Calculates quantiles for the given levels.

        Parameters
        ----------
        level
            Level values to use for computing the quantiles.
            `level` should be a 1d tensor of level values between 0 and 1.

        Returns
        -------
        quantiles
            Quantile values corresponding to the levels passed.
            The return shape is

               (num_levels, ...DISTRIBUTION_SHAPE...),

            where DISTRIBUTION_SHAPE is the shape of the underlying distribution.
        """
        raise NotImplementedError()

    def __getitem__(self, item):
        sliced_distr = self.__class__(
            *[_index_tensor(arg, item) for arg in self.args]
        )
        assert isinstance(sliced_distr, type(self))
        return sliced_distr

    def slice_axis(
        self, axis: int, begin: int, end: Optional[int]
    ) -> "Distribution":
        index: List[Any]
        if axis >= 0:
            index = [slice(None)] * axis + [slice(begin, end)]
        else:
            index = [Ellipsis, slice(begin, end)] + [slice(None)] * (-axis - 1)
        return self[tuple(index)]


def _expand_param(p: Tensor, num_samples: Optional[int] = None) -> Tensor:
    """
    Expand parameters by num_samples along the first dimension.
    """
    if num_samples is None:
        return p
    return p.expand_dims(axis=0).repeat(axis=0, repeats=num_samples)


def _sample_multiple(
    sample_func, *args, num_samples: Optional[int] = None, **kwargs
) -> Tensor:
    """
    Sample from the sample_func, by passing expanded args and kwargs and
    reshaping the returned samples afterwards.
    """
    args_expanded = [_expand_param(a, num_samples) for a in args]
    kwargs_expanded = {
        k: _expand_param(v, num_samples) for k, v in kwargs.items()
    }
    samples = sample_func(*args_expanded, **kwargs_expanded)
    return samples
