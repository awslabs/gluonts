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
    if np.ndim(time_series) == 1:
        if seasonality < np.size(time_series):
            forecast_freq = seasonality
        else:
            # edge case: the seasonal freq is larger than the length of ts
            forecast_freq = 1

        y_t = time_series[:-forecast_freq]
        y_tm = time_series[forecast_freq:]

        return np.mean(np.abs(y_t - y_tm))
    else:
        pass  # TODO: consider multivariate case


# TODO: make this simpler if possible
def expand_seaonal_error(
    seasonal_error_values: np.ndarray, target_shape: Tuple[int]
) -> np.ndarray:
    # add prediction_length axis to match dimension of forecasts

    if np.ndim(seasonal_error_values) == 1:
        # univariate case: (num_samples, prediction_length)
        values_with_added_dim = seasonal_error_values.reshape(-1, 1)
    elif np.ndim(seasonal_error_values) == 2:
        # multivariate case: (num_samples, prediction_length, target_dim)
        print(target_shape, seasonal_error_values.shape)
        x, _, z = target_shape
        values_with_added_dim = seasonal_error_values.reshape(x, 1, z)

    return np.broadcast_to(values_with_added_dim, target_shape)


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
    data: Dict[str, np.ndarray],
    alpha: float,
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

    return numerator / expand_seaonal_error(
        data["seasonal_error"], data["label"].shape
    )


def absolute_scaled_error(data: Dict[str, np.ndarray], forecast_type: str):
    return absolute_error(data, forecast_type) / expand_seaonal_error(
        data["seasonal_error"], data["label"].shape
    )
