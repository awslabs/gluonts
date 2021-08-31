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

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .distribution import Distribution, getF


class EmpiricalDistribution(Distribution):
    r"""
    A class representing empirical distribution.
    Multivariate target is also supported.

    Parameters
    ----------
    samples
        Tensor containing samples, of shape `(num_samples, *batch_shape, *event_shape)`.
    """

    @validated()
    def __init__(self, samples: Tensor, event_dim: int = 1) -> None:
        self.samples = samples
        self.sorted_samples = self.F.sort(self.samples, axis=0)
        self._event_dim = event_dim

    @property
    def F(self):
        return getF(self.samples)

    @property
    def batch_shape(self) -> Tuple:
        return self.samples.shape[1:-self.event_dim]

    @property
    def event_shape(self) -> Tuple:
        return self.samples.shape[-self.event_dim:]

    @property
    def event_dim(self) -> int:
        return self._event_dim

    @property
    def mean(self) -> Tensor:
        return self.F.mean(self.samples, axis=0)

    @property
    def stddev(self) -> Tensor:
        return self.F.std(self.samples, axis=0)

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        return self.samples

    def quantile(self, level: Tensor) -> Tensor:
        # `sample_idx` would be same for each element of the batch, time point and dimension.
        num_samples = self.sorted_samples.shape[0]
        sample_idx = np.round((num_samples - 1) * level)

        return self.sorted_samples[sample_idx, :, :]

    def quantile_losses(
        self, obs: Tensor, quantiles: Tensor, levels: Tensor
    ) -> Tensor:
        """
        Computes quantile losses for all the quantiles specified.

        Parameters
        ----------
        obs
            Ground truth observation. Shape: `(batch_size, seq_len, *event_shape)`
        quantiles
            Quantile values. Shape: `(batch_size, seq_len, *event_shape, num_quantiles)`
        levels
            Quantile levels. Shape: `(batch_size, seq_len, *event_shape, num_quantiles)`
        Returns
        -------
        Tensor
            Quantile losses of shape: `(batch_size, seq_len, *event_shape, num_quantiles)`

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

    def crps_univariate(self, obs: Tensor) -> Tensor:
        r"""
        Compute the *continuous rank probability score* (CRPS) of `obs` according
        to the empirical distribution.

        The last dimension of `obs` specifies the "event dimension" of the target (= 1 for the univariate case).
        For multivariate target, CRSP scores are computed for each dimension separately and then their sum is returned.

        Parameters
        ----------
        obs
            Tensor of ground truth with shape `(*batch_shape, *event_shape)`

        Returns
        -------
        Tensor
            CRPS score of shape `(*batch_shape, 1)`.
        """
        F = self.F
        sorted_samples = self.sorted_samples

        # We compute quantile losses for all the possible quantile levels; i.e, `num_quantiles` = `num_samples`.
        num_quantiles = sorted_samples.shape[0]
        levels_np = np.arange(
            1 / num_quantiles, 1 + 1 / num_quantiles, 1 / num_quantiles
        )

        # Quantiles are just sorted samples with a different shape (for convenience):
        # `(*batch_shape, *event_shape, num_quantiles)`
        axes_ordering = list(range(1, self.all_dim + 1))
        axes_ordering.append(0)
        quantiles = sorted_samples.transpose(axes_ordering)

        # Shape: `(*batch_shape, *event_shape, num_quantiles)`
        qlosses = self.quantile_losses(
            obs=obs,
            quantiles=quantiles,
            levels=F.array(levels_np).broadcast_like(quantiles),
        )

        # CRPS for each target dimension. Shape: `(*batch_shape, *event_shape)`
        crps = F.sum(qlosses, axis=-1)

        if self.event_dim > 0:
            # Total CRPS: sum over all but the axes corresponding to the batch shape.
            # Shape: `(*batch_shape)`
            crps = F.sum(crps, exclude=True, axis=list(range(0, len(self.batch_shape))))

        return crps.expand_dims(axis=-1)
