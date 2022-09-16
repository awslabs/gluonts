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

from typing import Collection, Union, Iterator

import numpy as np

from gluonts.dataset import DataEntry
from gluonts.model import Forecast
from gluonts.model.forecast import Quantile


class PrimitiveForecastBatch:
    def __init__(self, forecasts: Collection[Forecast]):
        self.forecasts = forecasts
        self.batch_size = len(forecasts)

    def quantile(self, q: Union[Quantile, float, str]) -> np.ndarray:
        result = [forecast.quantile(q) for forecast in self.forecasts]
        return np.stack(result)

    @property
    def median(self):
        return self.quantile(Quantile.parse(0.5))

    @property
    def mean(self) -> np.ndarray:
        result = [forecast.mean for forecast in self.forecasts]
        return np.stack(result)

    def __len__(self):
        return self.batch_size


def get_input_batches(
    dataset_it: Iterator[DataEntry],
    forecast_it: Iterator[Forecast],
    batch_size: int,
) -> Iterator[dict]:
    done = False
    while not done:
        target_batch = []
        past_data_batch = []
        forecast_batch = []

        try:
            for _ in range(batch_size):
                data_entry = next(dataset_it)
                forecast = next(forecast_it)

                target_batch.append(
                    data_entry["target"][-forecast.prediction_length :]
                )
                past_data_batch.append(
                    data_entry["target"][: -forecast.prediction_length]
                )
                forecast_batch.append(forecast)
        except StopIteration:
            done = True

        if len(target_batch) > 0:
            input_batch = {
                "target": np.stack(target_batch),
                "past_data": np.stack(past_data_batch),
                "forecast": PrimitiveForecastBatch(forecast_batch),
            }
            yield input_batch
