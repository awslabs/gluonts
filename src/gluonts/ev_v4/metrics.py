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

from typing import Optional, Collection

import numpy as np

from gluonts.exceptions import GluonTSUserError
from gluonts.time_feature import get_seasonality


# BASE METRICS (same shape as input)


def abs_label(data: dict):
    return np.abs(data["label"])


def error(data: dict, forecast_type: str = "mean"):
    return data["label"] - data[forecast_type]


def abs_error(data: dict, forecast_type: str = "mean"):
    return np.abs(error(data, forecast_type))


def squared_error(data: dict, forecast_type: str = "mean"):
    return np.square(error(data, forecast_type))


def quantile_loss(data: dict, q: float = 0.5):
    forecast_type = str(q)
    prediction = data[forecast_type]

    return np.abs(
        error(data, forecast_type) * ((prediction >= data["label"]) - q)
    )


def coverage(data: dict, q: float = 0.5):
    forecast_type = str(q)
    return data["label"] < data[forecast_type]


# TODO: check if ND makes sense without axis
def nd(data: dict, forecast_type: str = "0.5"):
    return abs_error(data, forecast_type) / abs_label(data)


# AGGREGATED METRICS


def abs_label_mean(data: dict, axis: Optional[int] = None):
    return np.mean(abs_label(data), axis=axis)


def mse(data: dict, forecast_type: str = "mean", axis: Optional[int] = None):
    return np.mean(squared_error(data, forecast_type), axis=axis)


def rmse(data: dict, forecast_type: str = "mean", axis: Optional[int] = None):
    return np.sqrt(mse(data, forecast_type, axis))


def nrmse(data: dict, forecast_type: str = "mean", axis: Optional[int] = None):
    return rmse(data, forecast_type, axis) / np.abs(data["label"])


def mape(data: dict, forecast_type: str = "0.5", axis: Optional[int] = None):
    return np.mean(abs_error(data, forecast_type) / abs_label(data), axis=axis)


def smape(data: dict, forecast_type: str = "0.5", axis: Optional[int] = None):
    return np.mean(
        abs_error(data, forecast_type)
        / (abs_label(data) + np.abs(data[forecast_type])),
        axis=axis,
    )


# TODO: try to make this less messy
def seasonal_error(
    data: dict,
    freq: Optional[str] = None,
    seasonality: Optional[int] = None,
    axis: Optional[int] = None,
):
    past_data = data["input"]

    if np.ndim(past_data) != 2:
        raise ValueError(
            "Seasonal error can't handle input data"
            " that is not 2-dimensional"
        )

    if not seasonality:
        if freq is None:
            raise GluonTSUserError(
                "To calculate the seasonal error, either "
                "`freq` or `seasonality` must be provided"
            )
        seasonality = get_seasonality(freq)

    if seasonality < np.shape(past_data)[axis]:
        forecast_freq = seasonality
    else:
        # edge case: the seasonal freq is larger than the length of ts
        forecast_freq = 1

    if axis == 0:
        y_t = past_data[:-forecast_freq, :]
        y_tm = past_data[forecast_freq:, :]
    elif axis == 1:
        y_t = past_data[:, :-forecast_freq]
        y_tm = past_data[:, forecast_freq:]
    else:
        raise ValueError(
            "Seasonal error can only handle 0 or 1 for axis argument"
        )

    return np.mean(np.abs(y_t - y_tm), axis=axis)


def mase(
    data: dict,
    forecast_type: str = "0.5",
    freq: Optional[str] = None,
    seasonality: Optional[int] = None,
    axis: Optional[int] = None,
) -> np.ndarray:
    return np.mean(
        np.abs(error(data, forecast_type)), axis=axis
    ) / seasonal_error(data, freq, seasonality, axis)


def msis(
    data: dict,
    alpha: float = 0.05,
    freq: Optional[str] = None,
    seasonality: Optional[int] = None,
    axis: Optional[int] = None,
):
    lower_quantile = data[str(alpha / 2)]
    upper_quantile = data[str(1.0 - alpha / 2)]

    label = data["label"]

    numerator = np.mean(
        upper_quantile
        - lower_quantile
        + 2.0 / alpha * (lower_quantile - label) * (label < lower_quantile)
        + 2.0 / alpha * (label - upper_quantile) * (label > upper_quantile),
        axis=axis,
    )

    return numerator / seasonal_error(data, freq, seasonality, axis)


def weighted_quantile_loss(
    data: dict, q: float = 0.5, axis: Optional[int] = None
):
    return np.sum(quantile_loss(data, q), axis=axis) / np.sum(
        abs_label(data), axis=axis
    )


def mae_coverage(
    data: dict,
    quantile_values: Collection[float] = (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ),
):
    return np.mean(
        [np.abs(np.mean(coverage(data, q) - q)) for q in quantile_values]
    )
