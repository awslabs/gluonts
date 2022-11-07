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

from dataclasses import dataclass, field
import logging
from functools import singledispatch
from typing import (
    Callable,
    Iterator,
    List,
    Optional,
    Any,
    Union,
    Type,
    Collection,
)

import numpy as np

from gluonts.core.component import tensor_to_numpy
from gluonts.dataset.common import DataEntry
from gluonts.model.forecast import (
    Forecast,
    QuantileForecast,
    SampleForecast,
    Quantile,
)


def fast_quantile(data: np.ndarray, q: float, axis: int) -> np.ndarray:
    """Fast Quantile, assumes that data is sorted along axis `axis`."""
    num_samples = data.shape[axis]

    idx, fraction = divmod(q * (num_samples - 1), 1)
    idx = int(idx)

    left = data.take(idx, axis=axis)
    if not fraction:
        return left

    right = data.take(idx + 1, axis=axis)

    return left + (right - left) * fraction


def _unpack(batched) -> Iterator:
    """
    Unpack batches.

    This assumes that arrays are wrapped in a  nested structure of lists and
    tuples, and each array has the same shape::

        >>> a = np.arange(5)
        >>> batched = [a, (a, [a, a, a])]
        >>> list(_unpack(batched))
        [[0, (0, [0, 0, 0])],
         [1, (1, [1, 1, 1])],
         [2, (2, [2, 2, 2])],
         [3, (3, [3, 3, 3])],
         [4, (4, [4, 4, 4])]]
    """

    if isinstance(batched, (list, tuple)):
        T = type(batched)

        return map(T, zip(*map(_unpack, batched)))

    return batched


@singledispatch
def make_distribution_forecast(distr, *args, **kwargs) -> Forecast:
    raise NotImplementedError


class ForecastBatch:
    @property
    def batches(self) -> Collection:
        raise NotImplementedError

    @property
    def batch_size(self) -> int:
        return len(self.batches)

    @property
    def mean(self) -> np.ndarray:
        raise NotImplementedError

    def quantile(self, q: Union[float, str]) -> np.ndarray:
        raise NotImplementedError


@dataclass
class DistributionForecastBatch(ForecastBatch):
    distr_args: list
    distr_output: Any  # TODO fix
    start: list
    item_id: Optional[list] = None
    info: Optional[list] = None
    distr: Type = field(init=False)

    @property
    def batches(self):
        return self.distr_args

    def __post_init__(self):
        self.distr = self.distr_output.distribution(*self.distr_args)
        if self.item_id is None:
            self.item_id = [None for _ in self.start]
        if self.info is None:
            self.info = [None for _ in self.start]

    def __iter__(self) -> Iterator[Forecast]:  # TODO fix
        distributions = [
            self.distr_output.distribution(*u)
            for u in _unpack(self.distr_args)
        ]

        for distr, start, item_id, info in zip(
            distributions, self.start, self.item_id, self.info
        ):
            yield make_distribution_forecast(
                distr,
                start_date=start,
                item_id=item_id,
                info=info,
            )

    @property
    def mean(self) -> np.ndarray:
        return tensor_to_numpy(self.distr.mean())

    def quantile(self, q: Union[float, str]) -> np.ndarray:
        return tensor_to_numpy(self.distr.quantile(q))


@dataclass
class SampleForecastBatch(ForecastBatch):
    samples: np.ndarray
    start: list
    item_id: Optional[list] = None
    info: Optional[list] = None
    sorted_samples: np.ndarray = field(init=False)

    @property
    def batches(self):
        return self.samples

    def __post_init__(self):
        self.sorted_samples = np.sort(self.samples, axis=1)

        if self.item_id is None:
            self.item_id = [None] * self.batch_size

        if self.info is None:
            self.info = [None] * self.batch_size

    def __iter__(self) -> Iterator[SampleForecast]:
        for sample, start, item_id, info in zip(
            self.samples, self.start, self.item_id, self.info
        ):
            yield SampleForecast(
                sample,
                start_date=start,
                item_id=item_id,
                info=info,
            )

    @property
    def num_samples(self) -> int:
        return self.samples.shape[1]

    @property
    def mean(self) -> np.ndarray:
        if self._mean is not None:
            return self._mean
        else:
            return np.mean(self.samples, axis=1)

    def quantile(self, q: Union[float, str]) -> np.ndarray:
        q: float = Quantile.parse(q).value
        return fast_quantile(self.sorted_samples, q, axis=1)


@dataclass
class QuantileForecastBatch(ForecastBatch):
    quantile_batch: np.ndarray
    quantile_levels: List[str]
    start: list
    item_id: Optional[list] = None
    info: Optional[list] = None

    @property
    def batches(self):
        return self.quantile_batch

    def __post_init__(self):
        if self.item_id is None:
            self.item_id = [None] * self.batch_size

        if self.info is None:
            self.info = [None] * self.batch_size

    def __iter__(self) -> Iterator[QuantileForecast]:
        for quantiles, start, item_id, info in zip(
            self.quantile_batch, self.start, self.item_id, self.info
        ):
            yield QuantileForecast(
                quantiles,
                start_date=start,
                item_id=item_id,
                info=info,
                forecast_keys=self.quantile_levels,
            )
