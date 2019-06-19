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
from typing import Tuple, List

# Third-party imports
import mxnet as mx
from mxnet import gluon

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor

# Relative imports
from .distribution import Distribution, _sample_multiple, getF
from .distribution_output import DistributionOutput


class Binned(Distribution):
    r"""
    A binned distribution defined by a set of bins via
    bin centers and bin probabilities.

    Parameters
    ----------
    bin_probs
        Tensor containing the bin probabilities, of shape `(*batch_shape, num_bins)`.
    bin_centers
        Tensor containing the bin centers, of shape `(*batch_shape, num_bins)`.
    F
    """

    is_reparameterizable = False

    def __init__(self, bin_probs: Tensor, bin_centers: Tensor, F=None) -> None:
        self.bin_centers = bin_centers
        self.bin_probs = bin_probs
        self.F = F if F else getF(bin_probs)

        self.bin_edges = Binned._compute_edges(self.F, bin_centers)

    @staticmethod
    def _compute_edges(F, bin_centers: Tensor) -> Tensor:
        r"""
        Computes the edges of the bins based on the centers. The first and last edge are set to :math:`10^{-10}` and
        :math:`10^{10}`, repsectively.

        Parameters
        ----------
        F
        bin_centers
            Tensor of shape `(*batch_shape, num_bins)`.

        Returns
        -------
        Tensor
            Tensor of shape (*batch.shape, num_bins+1)
        """

        low = (
            F.zeros_like(bin_centers.slice_axis(axis=-1, begin=0, end=1))
            - 1.0e10
        )
        high = (
            F.zeros_like(bin_centers.slice_axis(axis=-1, begin=0, end=1))
            + 1.0e10
        )

        means = (
            bin_centers.slice_axis(axis=-1, begin=1, end=None)
            + bin_centers.slice_axis(axis=-1, begin=0, end=-1)
        ) / 2.0

        return F.concat(low, means, high, dim=-1)

    @property
    def batch_shape(self) -> Tuple:
        return self.bin_centers.shape[:-1]

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    @property
    def mean(self):
        return (self.bin_probs * self.bin_centers).sum(axis=-1)

    @property
    def stddev(self):
        Ex2 = (self.bin_probs * self.bin_centers.square()).sum(axis=-1)
        return (Ex2 - self.mean.square()).sqrt()

    def log_prob(self, x):
        F = self.F
        x = x.expand_dims(axis=-1)
        # TODO: when mxnet has searchsorted replace this
        left_edges = self.bin_edges.slice_axis(axis=-1, begin=0, end=-1)
        right_edges = self.bin_edges.slice_axis(axis=-1, begin=1, end=None)
        mask = F.broadcast_lesser_equal(left_edges, x) * F.broadcast_lesser(
            x, right_edges
        )
        return F.broadcast_mul(self.bin_probs.log(), mask).sum(axis=-1)

    def cdf(self, x: Tensor) -> Tensor:
        F = self.F
        x = x.expand_dims(axis=-1)
        # left_edges = self.bin_edges.slice_axis(axis=-1, begin=0, end=-1)
        mask = F.broadcast_lesser_equal(self.bin_centers, x)
        return F.broadcast_mul(self.bin_probs, mask).sum(axis=-1)

    def sample(self, num_samples=None):
        def s(bin_probs):
            F = self.F
            indices = F.sample_multinomial(bin_probs)
            if num_samples is None:
                return self.bin_centers.pick(indices, -1).reshape_like(
                    F.zeros_like(indices.astype('float32'))
                )
            else:
                return F.repeat(
                    F.expand_dims(self.bin_centers, axis=0),
                    repeats=num_samples,
                    axis=0,
                ).pick(indices, -1)

        return _sample_multiple(s, self.bin_probs, num_samples=num_samples)


class BinnedArgs(gluon.HybridBlock):
    def __init__(self, num_bins: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_bins = num_bins
        with self.name_scope():
            # needs to be named self.proj for consistency with the
            # ArgProj class and the inference tests
            self.proj = gluon.nn.HybridSequential()
            self.proj.add(
                gluon.nn.Dense(
                    self.num_bins,
                    prefix='binproj',
                    flatten=False,
                    weight_initializer=mx.init.Xavier(),
                )
            )
            self.proj.add(gluon.nn.HybridLambda('softmax'))

    def hybrid_forward(self, F, x: Tensor, **kwargs) -> Tuple[Tensor]:
        ps = self.proj(x)
        return (ps.reshape(shape=(-2, -1, self.num_bins), reverse=1),)


class BinnedOutput(DistributionOutput):
    distr_cls: type = Binned

    @validated()
    def __init__(self, bin_centers: List[float]) -> None:
        self.bin_centers = mx.nd.array(bin_centers)
        self.num_bins = self.bin_centers.shape[0]
        assert len(self.bin_centers.shape) == 1

    def get_args_proj(self, *args, **kwargs) -> gluon.nn.HybridBlock:
        return BinnedArgs(self.num_bins)

    def distribution(self, args, scale=None) -> Binned:
        probs = args[0]
        F = getF(probs)

        bin_centers = F.broadcast_mul(self.bin_centers, F.ones_like(probs))

        if scale is not None:
            bin_centers = F.broadcast_mul(
                bin_centers, scale.expand_dims(axis=-1)
            )

        return Binned(probs, bin_centers)

    @property
    def event_shape(self) -> Tuple:
        return ()
