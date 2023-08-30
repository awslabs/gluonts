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

from typing import Optional, Tuple

import numpy as np
import mxnet as mx

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .distribution import Distribution, getF
from .distribution_output import DistributionOutput, ArgProj


class EmpiricalDistribution(Distribution):
    r"""
    A class representing empirical distribution.

    The target can be vector/tensor-valued, i.e., `event_shape` can be larger
    than or equal to 1. However, note that each dimension is assumed to be
    independent when computing variance and CRPS loss.

    Also, for computing CDF and quantiles, it is assumede that samples are
    distinct along the samples dimension, which should almost always be the
    case if samples are drawn from continuous distributions.

    Parameters
    ----------
    samples
        Tensor containing samples, of shape
        `(num_samples, *batch_shape, *event_shape)`.
    event_dim
        Number of event dimensions, i.e., length of the `event_shape` tuple.
        This is `0` for distributions over scalars, `1` over vectors,
        `2` over matrices, and so on.
    """

    @validated()
    def __init__(self, samples: Tensor, event_dim: int) -> None:
        self.samples = samples
        self.sorted_samples = self.F.sort(self.samples, axis=0)
        self.sorted_ix = self.F.argsort(self.samples, axis=0)

        self._event_dim = event_dim
        assert len(self.samples.shape) >= 1 + event_dim, (
            "Shape of samples do not match with the value given for"
            " `event_dim`!"
        )

    @property
    def F(self):
        return getF(self.samples)

    @property
    def batch_shape(self) -> Tuple:
        if self.event_dim == 0:
            return self.samples.shape[1:]
        else:
            return self.samples.shape[1 : -self.event_dim]

    @property
    def event_shape(self) -> Tuple:
        if self.event_dim == 0:
            return ()
        else:
            return self.samples.shape[-self.event_dim :]

    @property
    def event_dim(self) -> int:
        return self._event_dim

    @property
    def mean(self) -> Tensor:
        return self.F.mean(self.samples, axis=0)

    @property
    def stddev(self) -> Tensor:
        F = self.F
        return F.sqrt(F.mean(F.square(self.mean - self.samples), axis=0))

    def cdf(self, x: Tensor):
        # Note: computes CDF on each dimension of the target independently.

        ix = [
            np.searchsorted(arr, val)
            for arr, val in zip(self.sorted_samples.T.asnumpy(), x.asnumpy())
        ]

        CDF_sorted = self.F.linspace(
            start=1 / len(self.samples),
            stop=1,
            endpoint=True,
            num=len(self.samples),
        )
        return CDF_sorted.take(indices=mx.nd.array(ix), axis=0)

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        sample_idx = np.random.randint(
            low=0, high=self.samples.shape[0], size=num_samples
        )
        if num_samples is None:
            # Only one random sample is returned.
            return self.samples[sample_idx]

        return self.samples.take(indices=mx.nd.array(sample_idx), axis=0)

    def quantile(self, level: Tensor) -> Tensor:
        # Note: computes quantile on each dimension of the target
        # independently. `sample_idx` would be same for each element of the
        # batch, time point and dimension.
        num_samples = self.sorted_samples.shape[0]
        sample_idx = np.round(num_samples * level) - 1

        return self.sorted_samples[sample_idx, :]

    def quantile_losses(
        self, obs: Tensor, quantiles: Tensor, levels: Tensor
    ) -> Tensor:
        """
        Computes quantile losses for all the quantiles specified.

        Parameters
        ----------
        obs
            Ground truth observation. Shape:
            `(batch_size, seq_len, *event_shape)`
        quantiles
            Quantile values. Shape:
            `(batch_size, seq_len, *event_shape, num_quantiles)`
        levels
            Quantile levels. Shape:
            `(batch_size, seq_len, *event_shape, num_quantiles)`
        Returns
        -------
        Tensor
            Quantile losses of shape:
            `(batch_size, seq_len, *event_shape, num_quantiles)`
        """
        obs = obs.expand_dims(axis=-1)
        assert obs.shape[:-1] == quantiles.shape[:-1]
        assert obs.shape[:-1] == levels.shape[:-1]
        assert obs.shape[-1] == 1

        return self.F.where(
            obs >= quantiles,
            levels * (obs - quantiles),
            (1 - levels) * (quantiles - obs),
        )

    def crps_univariate(self, x: Tensor) -> Tensor:
        r"""Compute the *continuous rank probability score* (CRPS) of `obs`
        according to the empirical distribution.

        The last dimension of `obs` specifies the "event dimension" of the
        target (= 1 for the univariate case). For multivariate target, CRSP
        scores are computed for each dimension separately and then their sum
        is returned.

        Parameters
        ----------
        x
            Tensor of ground truth with shape `(*batch_shape, *event_shape)`

        Returns
        -------
        Tensor
            CRPS score of shape `(*batch_shape, 1)`.
        """
        F = self.F
        sorted_samples = self.sorted_samples

        # We compute quantile losses for all the possible quantile levels; i.e,
        # `num_quantiles` = `num_samples`.
        num_quantiles = sorted_samples.shape[0]
        levels_np = np.arange(
            1 / num_quantiles, 1 + 1 / num_quantiles, 1 / num_quantiles
        )

        # Quantiles are just sorted samples with a different shape
        # (for convenience): `(*batch_shape, *event_shape, num_quantiles)`
        axes_ordering = list(range(1, self.all_dim + 1))
        axes_ordering.append(0)
        quantiles = sorted_samples.transpose(axes_ordering)

        # Shape: `(*batch_shape, *event_shape, num_quantiles)`
        qlosses = self.quantile_losses(
            obs=x,
            quantiles=quantiles,
            levels=F.array(levels_np).broadcast_like(quantiles),
        )

        # CRPS for each target dimension. Shape: `
        # (*batch_shape, *event_shape)`
        crps = F.sum(qlosses, axis=-1)

        if self.event_dim > 0:
            # Total CRPS: sum over all but the axes corresponding to the batch
            # shape. Shape: `(*batch_shape)`
            crps = F.sum(
                crps, exclude=True, axis=list(range(0, len(self.batch_shape)))
            )

        return crps

    def loss(self, x: Tensor) -> Tensor:
        return self.crps_univariate(x=x)


class EmpiricalDistributionOutput(DistributionOutput):
    """
    This allows us to wrap `EmpiricalDistribution` by any parametric
    distribution and learn the parameters by minimizing CRPS loss on the
    samples of `EmpiricalDistribution`.

    See the inference test `test_empirical_distribution` in
    `test.distribution.test_mx_distribution_inference` which checks if the CRPS
    loss is correctly implemented.
    """

    @validated()
    def __init__(
        self,
        num_samples: int,
        distr_output: DistributionOutput,
    ) -> None:
        super().__init__(self)
        self.num_samples = num_samples
        self.distr_output = distr_output

    def get_args_proj(self, prefix: Optional[str] = None) -> ArgProj:
        return self.distr_output.get_args_proj(prefix=prefix)

    def distribution(
        self,
        distr_args,
        loc: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ) -> Distribution:
        distr = self.distr_output.distribution(
            distr_args=distr_args, loc=loc, scale=scale
        )

        # Here `sample_rep` should be differentiable!
        samples = distr.sample_rep(num_samples=self.num_samples)

        return EmpiricalDistribution(
            samples=samples, event_dim=distr.event_dim
        )

    def domain_map(self, F, *args, **kwargs):
        return self.distr_output.domain_map(F, *args, **kwargs)

    @property
    def event_shape(self) -> Tuple:
        return self.distr_output.event_shape
