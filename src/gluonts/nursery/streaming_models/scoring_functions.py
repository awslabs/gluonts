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

from typing import List, Tuple

import mxnet as mx
import numpy as np
import pandas as pd

from gluonts.core.component import validated
from gluonts.model.forecast import Forecast
from gluonts.mx.distribution import (
    Binned,
    Distribution,
    TransformedDistribution,
)
from gluonts.mx.model.forecast import DistributionForecast

NO_SCORE = np.nan


def tail_probability_binned(
    distr: Distribution,
    x: mx.nd.NDArray,
    only_lower_tail: bool,
    only_upper_tail: bool,
) -> mx.nd.NDArray:
    if isinstance(distr, TransformedDistribution):
        for t in distr.transforms[::-1]:
            x = t.f_inv(x)
        distr = distr.base_distribution

    assert isinstance(distr, Binned)

    F = mx.nd
    x = x.expand_dims(axis=-1)
    bcs = distr.bin_centers
    bps = distr.bin_probs
    mask_upper = F.broadcast_greater_equal(bcs, x)
    p_upper_tail = F.broadcast_mul(bps, mask_upper).sum(axis=-1).asnumpy()
    if only_upper_tail:
        return p_upper_tail
    mask_lower = F.broadcast_lesser_equal(bcs, x)
    p_lower_tail = F.broadcast_mul(bps, mask_lower).sum(axis=-1).asnumpy()
    if only_lower_tail:
        return p_lower_tail

    # cap to 0.5 to avoid degenerate cases when the bin that x belongs has high probability
    p_extreme = np.minimum(np.minimum(p_lower_tail, p_upper_tail), 0.5)
    return p_extreme


def rarity(binned_distr: Binned, x: mx.nd.NDArray) -> mx.nd.NDArray:
    """
    Computes the _rarity_ score
        rarity(x) = - log P{p(X) <= p(x)}

    TODO: Move this to Binned in GluonTS. Use log_bin_probs and do logsumexp.
    """
    assert isinstance(binned_distr, Binned)

    F = mx.nd
    x = x.expand_dims(axis=-1)
    left_edges = binned_distr.bin_edges.slice_axis(axis=-1, begin=0, end=-1)
    right_edges = binned_distr.bin_edges.slice_axis(axis=-1, begin=1, end=None)
    mask = F.broadcast_lesser_equal(left_edges, x) * F.broadcast_lesser(
        x, right_edges
    )
    pdf = F.broadcast_mul(binned_distr.bin_probs, mask).sum(axis=-1)
    lesser_mask = F.broadcast_lesser_equal(
        binned_distr.bin_probs, pdf.expand_dims(axis=-1)
    )
    prod = F.broadcast_mul(lesser_mask, binned_distr.bin_probs)
    return F.maximum(-F.log10(F.sum(prod, axis=-1).clip(1e-10, 1)), 0)


class ScoringFunction:
    """
    An abstract class representing a scoring function as a serializable object.

    Any implementation of this class should implement the `_compute_scores`
    method. The entry point to use the scoring function is the `__call__`
    method, which conveniently wraps `_compute_scores` with some assertions
    and back-filling logic.
    """

    @validated()
    def __init__(self, *args, **kwargs) -> None:
        pass

    def _compute_scores(
        self, forecast: Forecast, observations: List[float]
    ) -> List[float]:
        """
        Compute a score for each observation, given the forecast.

        Parameters
        ----------
        forecast
            Forecast object which should be used to score the observations.
        observations
            Observations to be scored.

        Returns
        -------
        List[float]
            List of scores, one per observation.
        """
        raise NotImplementedError()

    def __call__(self, forecast_data: Tuple[Forecast, pd.Series]) -> pd.Series:
        forecast, data = forecast_data

        assert forecast.start_date >= data.start

        backfill = int(
            (forecast.start_date - data.start) / pd.Timedelta(data.start.freq)
        )

        observations = data.to_list()[backfill:]

        computed_scores = [
            float(s) if not np.isnan(s) else NO_SCORE
            for s in self._compute_scores(
                forecast=forecast, observations=observations
            )
        ]

        assert len(computed_scores) == len(observations)

        scores = [NO_SCORE for _ in range(backfill)] + computed_scores

        nan_mask = np.isnan(data.target)

        # Until we properly model the probability of nan, we return NO_SCORE
        scores_array = np.asarray(scores, dtype=np.float32)
        scores_array[nan_mask] = NO_SCORE

        return pd.Series(scores_array, index=data.index)


class TailProbability(ScoringFunction):
    def __init__(
        self, only_lower_tail: bool = False, only_upper_tail: bool = False
    ):
        super().__init__()
        assert not (
            only_lower_tail and only_upper_tail
        ), "Both `only_lower_tail` and `only_upper_tail` cannot be True!"
        self.only_lower_tail = only_lower_tail
        self.only_upper_tail = only_upper_tail

    def cdf_to_tail_prob(self, cdf_obs) -> np.ndarray:
        if self.only_upper_tail:
            return 1.0 - cdf_obs
        elif self.only_lower_tail:
            return cdf_obs
        else:
            return np.minimum(cdf_obs, 1.0 - cdf_obs)

    def _compute_scores(
        self, forecast: Forecast, observations: List[float]
    ) -> List[float]:
        if isinstance(forecast, DistributionForecast):
            observations = mx.nd.array(observations)
            distr = forecast.distribution
            if isinstance(distr, Binned) or (
                isinstance(distr, TransformedDistribution)
                and isinstance(distr.base_distribution, Binned)
            ):
                p_extreme = tail_probability_binned(
                    distr,
                    observations,
                    only_lower_tail=self.only_lower_tail,
                    only_upper_tail=self.only_upper_tail,
                )
            else:
                cdf_obs = forecast.distribution.cdf(observations).asnumpy()
                p_extreme = self.cdf_to_tail_prob(cdf_obs)
        else:
            raise NotImplementedError("unsupported forecast type")

        return list(-np.log10(np.maximum(1e-10, p_extreme)))


class Rarity(ScoringFunction):
    def _compute_scores(
        self, forecast: Forecast, observations: List[float]
    ) -> List[float]:
        assert isinstance(forecast, DistributionForecast)
        assert isinstance(forecast.distribution, Binned)

        return list(rarity(forecast.distribution, mx.nd.array(observations)))
