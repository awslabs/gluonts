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

from typing import Dict, Tuple

import numpy as np


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


def absolute_label(data: Dict[str, np.ndarray]) -> np.ndarray:
    return np.abs(data["label"])


def error(data: Dict[str, np.ndarray], forecast_type: str) -> np.ndarray:
    return data["label"] - data[forecast_type]


def absolute_error(
    data: Dict[str, np.ndarray], forecast_type: str
) -> np.ndarray:
    return np.abs(error(data, forecast_type))


def squared_error(
    data: Dict[str, np.ndarray], forecast_type: str
) -> np.ndarray:
    return np.square(error(data, forecast_type))


def quantile_loss(data: Dict[str, np.ndarray], q: float) -> np.ndarray:
    forecast_type = str(q)
    prediction = data[forecast_type]

    return np.abs(
        error(data, forecast_type) * ((prediction >= data["label"]) - q)
    )


def coverage(data: Dict[str, np.ndarray], q: float) -> np.ndarray:
    forecast_type = str(q)
    return data["label"] < data[forecast_type]


def absolute_percentage_error(
    data: Dict[str, np.ndarray], forecast_type: str
) -> np.ndarray:
    return absolute_error(data, forecast_type) / absolute_label(data)


def symmetric_absolute_percentage_error(
    data: Dict[str, np.ndarray], forecast_type: str
) -> np.ndarray:
    return absolute_error(data, forecast_type) / (
        absolute_label(data) + np.abs(data[forecast_type])
    )


def scaled_interval_score(
    data: Dict[str, np.ndarray], alpha: float
) -> np.ndarray:
    lower_quantile = data[str(alpha / 2)]
    upper_quantile = data[str(1.0 - alpha / 2)]
    label = data["label"]

    numerator = (
        upper_quantile
        - lower_quantile
        + 2.0 / alpha * (lower_quantile - label) * (label < lower_quantile)
        + 2.0 / alpha * (label - upper_quantile) * (label > upper_quantile)
    )

    return numerator / data["seasonal_error"]


def absolute_scaled_error(data: Dict[str, np.ndarray], forecast_type: str):
    return absolute_error(data, forecast_type) / data["seasonal_error"]
