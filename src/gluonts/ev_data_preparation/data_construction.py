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

from typing import ChainMap, Dict, Iterator

import numpy as np
from gluonts.model.forecast import SampleForecast

from gluonts.time_feature.seasonality import get_seasonality
from gluonts.model.predictor import Predictor
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
    test_data: TestData, predictor: Predictor, **predictor_kwargs
) -> Iterator[Dict[str, np.ndarray]]:
    forecasts = predictor.predict(dataset=test_data.input, **predictor_kwargs)

    for input, label, forecast in zip(
        test_data.input, test_data.label, forecasts
    ):
        batching_used = isinstance(forecast, ForecastBatch)

        non_forecast_data = {
            "label": label["target"],
            "seasonal_error": seasonal_error(
                input["target"],
                seasonality=get_seasonality(freq=forecast.start_date.freqstr),
            ),
        }

        if batching_used:
            forecast_data = forecast
        else:
            non_forecast_data = {
                key: np.array([value])
                for key, value in non_forecast_data.items()
            }

            if isinstance(forecast, SampleForecast):
                forecast_data = SampleForecastBatch.from_forecast(forecast)
            else:
                pass  # TODO

        joint_data = ChainMap(non_forecast_data, forecast_data)

        yield joint_data
