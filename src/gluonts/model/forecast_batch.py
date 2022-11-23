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

from dataclasses import dataclass
from typing import Iterator, List, Optional, Union

import numpy as np

from .forecast import Forecast, Quantile, QuantileForecast, SampleForecast

# TODO: replace this with code from PR #2286 when it's ready


class ForecastBatch:
    # ...

    @classmethod
    def from_forecast(cls, forecast: Forecast) -> "ForecastBatch":
        """Returns a new `ForecastBatch` with batch dimension 1"""
        raise NotImplementedError

    def __getitem__(self, name):
        if name == "mean":
            return self.mean
        elif name == "median":
            return self.median

        return self.quantile(name)

    # ...


@dataclass
class SampleForecastBatch(ForecastBatch):
    samples: np.ndarray
    start_date: list
    item_id: Optional[list] = None
    info: Optional[list] = None

    def __post_init__(self):
        self._sorted_samples_value = None
        if self.item_id is None:
            self.item_id = [None for _ in self.start_date]
        if self.info is None:
            self.info = [None for _ in self.start_date]

    @property
    def _sorted_samples(self) -> np.ndarray:
        if self._sorted_samples_value is None:
            self._sorted_samples_value = np.sort(self.samples, axis=1)
        return self._sorted_samples_value

    def __iter__(self) -> Iterator[SampleForecast]:
        for sample, start_date, item_id, info in zip(
            self.samples, self.start_date, self.item_id, self.info
        ):
            yield SampleForecast(
                sample,
                start_date=start_date,
                item_id=item_id,
                info=info,
            )

    @property
    def batch_size(self) -> int:
        return self.samples.shape[0]

    @property
    def num_samples(self) -> int:
        return self.samples.shape[1]

    @property
    def mean(self) -> np.ndarray:
        return np.mean(self.samples, axis=1)

    def quantile(self, q: Union[float, str]) -> np.ndarray:
        q = Quantile.parse(q).value
        sample_idx = int(np.round((self.num_samples - 1) * q))
        return self._sorted_samples[:, sample_idx, :]

    @classmethod
    def from_forecast(cls, forecast: SampleForecast) -> "SampleForecastBatch":
        return SampleForecastBatch(
            samples=np.array([forecast.samples]),
            start_date=[forecast.start_date],
            item_id=[forecast.item_id],
            info=[forecast.info],
        )


@dataclass
class QuantileForecastBatch(ForecastBatch):
    quantile_batch: np.ndarray
    quantile_levels: List[str]
    start: list
    item_id: Optional[list] = None
    info: Optional[list] = None

    def __post_init__(self):
        if self.item_id is None:
            self.item_id = [None for _ in self.start]
        if self.info is None:
            self.info = [None for _ in self.start]

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

    @classmethod
    def from_forecast(
        cls, forecast: QuantileForecast
    ) -> "QuantileForecastBatch":
        return QuantileForecastBatch(
            forecast_arrays=np.array([forecast.forecast_arrays]),
            forecast_keys=[forecast.forecast_keys],
            start_date=[forecast.start_date],
            item_id=[forecast.item_id],
            info=[forecast.info],
        )
