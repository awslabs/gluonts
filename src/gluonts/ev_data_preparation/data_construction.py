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

from typing import ChainMap, Dict, Iterator, Union

import numpy as np
from gluonts.model.forecast import Forecast, SampleForecast

from gluonts.time_feature.seasonality import get_seasonality
from gluonts.dataset.split import TestData
from .forecast_batch import ForecastBatch, SampleForecastBatch


def seasonal_error(time_series: np.ndarray, seasonality: int) -> np.ndarray:
    """The seasonal error is a the mean absolute difference of a given time
    series, shifted by its seasonality.

    Some metrics use the seasonal error for normalization."""

    if seasonality < time_series.shape[-1]:
        forecast_freq = seasonality
    else:
        # edge case: the seasonal freq is larger than the length of ts
        forecast_freq = 1

    if time_series.ndim == 1:
        y_t = time_series[:-forecast_freq]
        y_tm = time_series[forecast_freq:]

        return np.abs(y_t - y_tm).mean()
    else:
        # multivariate case:
        # time_series is has shape (# of time stamps, # of variates)
        y_t = time_series[:-forecast_freq, :]
        y_tm = time_series[forecast_freq:, :]

        return np.abs(y_t - y_tm).mean(axis=0, keep_dims=True)


def construct_data(
    test_data: TestData,
    forecasts: Iterator[Union[Forecast, ForecastBatch]],
    ignore_invalid_values: bool = True,
) -> Iterator[Dict[str, np.ndarray]]:
    """construct data for evaluation

    ignore_invalid_values
        Ignore `NaN` and `inf` values in the timeseries when calculating
        metrics, defaults to True
    """
    for input, label, forecast in zip(
        test_data.input, test_data.label, forecasts
    ):
        batching_used = isinstance(forecast, ForecastBatch)

        input_target = input["target"]
        label_target = label["target"]

        if ignore_invalid_values:
            input_target = np.ma.masked_invalid(input_target)
            label_target = np.ma.masked_invalid(label_target)

        non_forecast_data = {
            "label": label_target,
            "seasonal_error": seasonal_error(
                input_target,
                seasonality=get_seasonality(freq=forecast.start_date.freqstr),
            ),
        }

        if batching_used:
            forecast_data = forecast
        else:
            non_forecast_data = {
                key: np.expand_dims(value, axis=0)
                for key, value in non_forecast_data.items()
            }

            if isinstance(forecast, SampleForecast):
                forecast_data = SampleForecastBatch.from_forecast(forecast)
            else:
                pass  # TODO

        joint_data = ChainMap(non_forecast_data, forecast_data)

        yield joint_data
