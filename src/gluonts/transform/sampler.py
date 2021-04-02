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

from typing import Tuple

import numpy as np
from pydantic import BaseModel

from gluonts.core.component import validated
from gluonts.dataset.stat import ScaleHistogram


class InstanceSampler(BaseModel):
    """
    An InstanceSampler is called with the time series ``ts``, and returns
    a set of indices at which training instances will be generated.

    The sampled indices ``i`` satisfy ``a <= i <= b``, where ``a = min_past``
    and ``b = ts.shape[axis] - min_future``.
    """

    axis: int = -1
    min_past: int = 0
    min_future: int = 0

    class Config:
        arbitrary_types_allowed = True

    def _get_bounds(self, ts: np.ndarray) -> Tuple[int, int]:
        return (
            self.min_past,
            ts.shape[self.axis] - self.min_future,
        )

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class UniformSplitSampler(InstanceSampler):
    """
    Samples each point with the same fixed probability.

    Parameters
    ----------
    p
        Probability of selecting a time point
    """

    p: float

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)

        if a > b:
            return np.array([], dtype=int)

        window_size = b - a + 1
        (indices,) = np.where(np.random.random_sample(window_size) < self.p)
        return indices + a


class PredictionSplitSampler(InstanceSampler):
    """
    Sampler used for prediction. Always selects the last time point for
    splitting i.e. the forecast point for the time series.
    """

    allow_empty_interval: bool = False

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        assert self.allow_empty_interval or a <= b
        return np.array([b]) if a <= b else np.array([], dtype=int)


def ValidationSplitSampler(
    axis: int = -1, min_past: int = 0, min_future: int = 0
) -> PredictionSplitSampler:
    return PredictionSplitSampler(
        allow_empty_interval=True,
        axis=axis,
        min_past=min_past,
        min_future=min_future,
    )


def TestSplitSampler(
    axis: int = -1, min_past: int = 0
) -> PredictionSplitSampler:
    return PredictionSplitSampler(
        allow_empty_interval=False,
        axis=axis,
        min_past=min_past,
        min_future=0,
    )


class ExpectedNumInstanceSampler(InstanceSampler):
    """
    Keeps track of the average time series length and adjusts the probability
    per time point such that on average `num_instances` training examples are
    generated per time series.

    Parameters
    ----------

    num_instances
        number of training examples generated per time series on average
    """

    num_instances: float
    total_length: int = 0
    n: int = 0

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1

        if window_size <= 0:
            return np.array([], dtype=int)

        self.n += 1
        self.total_length += window_size
        avg_length = self.total_length / self.n

        if avg_length <= 0:
            return np.array([], dtype=int)

        p = self.num_instances / avg_length
        (indices,) = np.where(np.random.random_sample(window_size) < p)
        return indices + a


class BucketInstanceSampler(InstanceSampler):
    """
    This sample can be used when working with a set of time series that have a
    skewed distributions. For instance, if the dataset contains many time series
    with small values and few with large values.

    The probability of sampling from bucket i is the inverse of its number of elements.

    Parameters
    ----------
    scale_histogram
        The histogram of scale for the time series. Here scale is the mean abs
        value of the time series.
    """

    scale_histogram: ScaleHistogram

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        p = 1.0 / self.scale_histogram.count(ts)
        (indices,) = np.where(np.random.random_sample(b - a + 1) < p)
        return indices + a


class ContinuousTimePointSampler(BaseModel):
    """
    Abstract class for "continuous time" samplers, which, given a lower bound
    and upper bound, sample "points" (events) in continuous time from a
    specified interval.
    """

    min_past: float = 0.0
    min_future: float = 0.0

    def _get_bounds(self, interval_length: float) -> Tuple[float, float]:
        return (
            self.min_past,
            interval_length - self.min_future,
        )

    def __call__(self, interval_length: float) -> np.ndarray:
        """
        Returns random points in the real interval between :code:`a` and
        :code:`b`.

        Parameters
        ----------
        a
            The lower bound (minimum time value that a sampled point can take)
        b
            Upper bound. Must be greater than a.
        """
        raise NotImplementedError()


class ContinuousTimeUniformSampler(ContinuousTimePointSampler):
    """
    Implements a simple random sampler to sample points in the continuous
    interval between :code:`a` and :code:`b`.
    """

    num_instances: int

    def __call__(self, interval_length: float) -> np.ndarray:
        a, b = self._get_bounds(interval_length)
        return (
            np.random.rand(self.num_instances) * (b - a) + a
            if a <= b
            else np.array([])
        )


class ContinuousTimePredictionSampler(ContinuousTimePointSampler):
    allow_empty_interval: bool = False

    def __call__(self, interval_length: float) -> np.ndarray:
        a, b = self._get_bounds(interval_length)
        assert (
            self.allow_empty_interval or a <= b
        ), "Interval start time must be before interval end time."
        return np.array([b]) if a <= b else np.array([])
