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

from typing import List, Optional, Tuple

import mxnet as mx
import numpy as np
from mxnet import gluon

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .distribution import MAX_SUPPORT_VAL, Distribution, _sample_multiple, getF
from .distribution_output import DistributionOutput


class Binned(Distribution):
    r"""
    A binned distribution defined by a set of bins via
    bin centers and bin probabilities.

    Parameters
    ----------
    bin_log_probs
        Tensor containing log probabilities of the bins, of shape
        `(*batch_shape, num_bins)`.
    bin_centers
        Tensor containing the bin centers, of shape `(*batch_shape, num_bins)`.
    F
    label_smoothing
        The label smoothing weight, real number in `[0, 1)`. Default `None`. If not
        `None`, then the loss of the distribution will be "label smoothed" cross-entropy.
        For example, instead of computing cross-entropy loss between the estimated bin
        probabilities and a hard-label (one-hot encoding) `[1, 0, 0]`, a soft label of
        `[0.9, 0.05, 0.05]` is taken as the ground truth (when `label_smoothing=0.15`).
        See (Muller et al., 2019) [MKH19]_, for further reference.
    """

    is_reparameterizable = False

    @validated()
    def __init__(
        self,
        bin_log_probs: Tensor,
        bin_centers: Tensor,
        label_smoothing: Optional[float] = None,
    ) -> None:
        self.bin_centers = bin_centers
        self.bin_log_probs = bin_log_probs
        self._bin_probs = None

        self.bin_edges = Binned._compute_edges(self.F, bin_centers)
        self.label_smoothing = label_smoothing

    @property
    def F(self):
        return getF(self.bin_log_probs)

    @property
    def support_min_max(self) -> Tuple[Tensor, Tensor]:
        F = self.F
        return (
            F.broadcast_minimum(
                F.zeros(self.batch_shape),
                F.sign(F.min(self.bin_centers, axis=-1)),
            )
            * MAX_SUPPORT_VAL,
            F.broadcast_maximum(
                F.zeros(self.batch_shape),
                F.sign(F.max(self.bin_centers, axis=-1)),
            )
            * MAX_SUPPORT_VAL,
        )

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
            F.broadcast_add(
                bin_centers.slice_axis(axis=-1, begin=1, end=None),
                bin_centers.slice_axis(axis=-1, begin=0, end=-1),
            )
            / 2.0
        )

        return F.concat(low, means, high, dim=-1)

    @property
    def bin_probs(self):
        if self._bin_probs is None:
            self._bin_probs = self.bin_log_probs.exp()
        return self._bin_probs

    @property
    def batch_shape(self) -> Tuple:
        return self.bin_log_probs.shape[:-1]

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    @property
    def mean(self):
        F = self.F
        return F.broadcast_mul(self.bin_probs, self.bin_centers).sum(axis=-1)

    @property
    def stddev(self):
        F = self.F
        ex2 = F.broadcast_mul(self.bin_probs, self.bin_centers.square()).sum(
            axis=-1
        )
        return F.broadcast_minus(ex2, self.mean.square()).sqrt()

    def _get_mask(self, x):
        F = self.F
        # TODO: when mxnet has searchsorted replace this
        left_edges = self.bin_edges.slice_axis(axis=-1, begin=0, end=-1)
        right_edges = self.bin_edges.slice_axis(axis=-1, begin=1, end=None)
        mask = F.broadcast_mul(
            F.broadcast_lesser_equal(left_edges, x),
            F.broadcast_lesser(x, right_edges),
        )
        return mask

    @staticmethod
    def _smooth_mask(F, mask, alpha):
        return F.broadcast_add(
            F.broadcast_mul(mask, F.broadcast_sub(F.ones_like(alpha), alpha)),
            F.broadcast_mul(F.softmax(F.ones_like(mask)), alpha),
        )

    def smooth_ce_loss(self, x):
        """
        Cross-entropy loss with a "smooth" label.
        """
        assert self.label_smoothing is not None
        F = self.F
        x = x.expand_dims(axis=-1)
        mask = self._get_mask(x)

        alpha = F.full(shape=(1,), val=self.label_smoothing)
        smooth_mask = self._smooth_mask(F, mask, alpha)

        return -F.broadcast_mul(self.bin_log_probs, smooth_mask).sum(axis=-1)

    def log_prob(self, x):
        F = self.F
        x = x.expand_dims(axis=-1)
        mask = self._get_mask(x)
        return F.broadcast_mul(self.bin_log_probs, mask).sum(axis=-1)

    def cdf(self, x: Tensor) -> Tensor:
        F = self.F
        x = x.expand_dims(axis=-1)
        # left_edges = self.bin_edges.slice_axis(axis=-1, begin=0, end=-1)
        mask = F.broadcast_lesser_equal(self.bin_centers, x)
        return F.broadcast_mul(self.bin_probs, mask).sum(axis=-1)

    def loss(self, x: Tensor) -> Tensor:
        return (
            self.smooth_ce_loss(x)
            if self.label_smoothing
            else -self.log_prob(x)
        )

    def quantile(self, level: Tensor) -> Tensor:
        F = self.F

        # self.bin_probs.shape = (batch_shape, num_bins)
        probs = self.bin_probs.transpose()  # (num_bins, batch_shape.T)

        # (batch_shape)
        zeros_batch_size = F.zeros_like(
            F.slice_axis(self.bin_probs, axis=-1, begin=0, end=1).squeeze(
                axis=-1
            )
        )

        level = level.expand_dims(axis=0)

        # cdf shape (batch_size.T, levels)
        zeros_cdf = F.broadcast_add(
            zeros_batch_size.transpose().expand_dims(axis=-1),
            level.zeros_like(),
        )
        start_state = (zeros_cdf, zeros_cdf.astype("int32"))

        def step(p, state):
            cdf, idx = state
            cdf = F.broadcast_add(cdf, p.expand_dims(axis=-1))
            idx = F.where(F.broadcast_greater(cdf, level), idx, idx + 1)
            return zeros_batch_size, (cdf, idx)

        _, states = F.contrib.foreach(step, probs, start_state)
        _, idx = states

        # idx.shape = (batch.T, levels)
        # centers.shape = (batch, num_bins)
        #
        # expand centers to shape -> (levels, batch, num_bins)
        # so we can use pick with idx.T.shape = (levels, batch)
        #
        # zeros_cdf.shape (batch.T, levels)
        centers_expanded = F.broadcast_add(
            self.bin_centers.transpose().expand_dims(axis=-1),
            zeros_cdf.expand_dims(axis=0),
        ).transpose()

        # centers_expanded.shape = (levels, batch, num_bins)
        # idx.shape (batch.T, levels)
        a = centers_expanded.pick(idx.transpose(), axis=-1)
        return a

    def sample(self, num_samples=None, dtype=np.float32):
        def s(bin_probs):
            F = self.F
            indices = F.sample_multinomial(bin_probs)
            if num_samples is None:
                return self.bin_centers.pick(indices, -1).reshape_like(
                    F.zeros_like(indices.astype("float32"))
                )
            else:
                return F.repeat(
                    F.expand_dims(self.bin_centers, axis=0),
                    repeats=num_samples,
                    axis=0,
                ).pick(indices, -1)

        return _sample_multiple(s, self.bin_probs, num_samples=num_samples)

    @property
    def args(self) -> List:
        return [self.bin_log_probs, self.bin_centers]


class BinnedArgs(gluon.HybridBlock):
    def __init__(
        self, num_bins: int, bin_centers: mx.nd.NDArray, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.num_bins = num_bins
        with self.name_scope():
            self.bin_centers = self.params.get_constant(
                "bin_centers", bin_centers
            )

            # needs to be named self.proj for consistency with the
            # ArgProj class and the inference tests
            self.proj = gluon.nn.HybridSequential()
            self.proj.add(
                gluon.nn.Dense(
                    self.num_bins,
                    prefix="binproj",
                    flatten=False,
                    weight_initializer=mx.init.Xavier(),
                )
            )
            self.proj.add(gluon.nn.HybridLambda("log_softmax"))

    def hybrid_forward(
        self, F, x: Tensor, bin_centers: Tensor
    ) -> Tuple[Tensor, Tensor]:
        ps = self.proj(x)
        reshaped_probs = ps.reshape(shape=(-2, -1, self.num_bins), reverse=1)
        bin_centers = F.broadcast_add(bin_centers, ps.zeros_like())
        return reshaped_probs, bin_centers


class BinnedOutput(DistributionOutput):
    distr_cls: type = Binned

    @validated()
    def __init__(
        self,
        bin_centers: mx.nd.NDArray,
        label_smoothing: Optional[float] = None,
    ) -> None:
        assert label_smoothing is None or (
            0 <= label_smoothing < 1
        ), "Smoothing factor should be less than 1 and greater than or equal to 0."
        super().__init__(self)
        self.bin_centers = bin_centers
        self.num_bins = self.bin_centers.shape[0]
        self.label_smoothing = label_smoothing
        assert len(self.bin_centers.shape) == 1

    def get_args_proj(self, *args, **kwargs) -> gluon.nn.HybridBlock:
        return BinnedArgs(self.num_bins, self.bin_centers)

    @staticmethod
    def _scale_bin_centers(F, bin_centers, loc=None, scale=None):
        if scale is not None:
            bin_centers = F.broadcast_mul(
                bin_centers, scale.expand_dims(axis=-1)
            )
        if loc is not None:
            bin_centers = F.broadcast_add(
                bin_centers, loc.expand_dims(axis=-1)
            )

        return bin_centers

    def distribution(self, args, loc=None, scale=None) -> Binned:
        probs = args[0]
        bin_centers = args[1]
        F = getF(probs)

        bin_centers = F.broadcast_mul(bin_centers, F.ones_like(probs))
        bin_centers = self._scale_bin_centers(
            F, bin_centers, loc=loc, scale=scale
        )

        return Binned(probs, bin_centers, label_smoothing=self.label_smoothing)

    @property
    def event_shape(self) -> Tuple:
        return ()
