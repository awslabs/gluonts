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

import numpy as np

from gluonts.ev.helpers import EvalData
from gluonts.exceptions import GluonTSUserError


def abs_label(data: EvalData) -> np.ndarray:
    return np.abs(data["label"])


def error(data: EvalData, forecast_type: str) -> np.ndarray:
    return data["label"] - data[forecast_type]


def abs_error(data: EvalData, forecast_type: str) -> np.ndarray:
    return np.abs(error(data, forecast_type))


def squared_error(data: EvalData, forecast_type: str) -> np.ndarray:
    return np.square(error(data, forecast_type))


def quantile_loss(data: EvalData, q: float) -> np.ndarray:
    forecast_type = str(q)
    prediction = data[forecast_type]

    return np.abs(
        error(data, forecast_type) * ((prediction >= data["label"]) - q)
    )


def coverage(data: EvalData, q: float) -> np.ndarray:
    forecast_type = str(q)
    return data["label"] < data[forecast_type]


def absolute_percentage_error(
    data: EvalData, forecast_type: str
) -> np.ndarray:
    return abs_error(data, forecast_type) / abs_label(data)


def symmetric_absolute_percentage_error(
    data: EvalData, forecast_type: str
) -> np.ndarray:
    return abs_error(data, forecast_type) / (
        abs_label(data) + np.abs(data[forecast_type])
    )


def seasonal_error_without_mean(
    data: dict,
    seasonality: int,
):
    """Calculates the seasonal error without applying the mean at the end.

    Further calculation to get the true metric value happens in
    `SeasonalError`.
    """
    past_data = data["input"]

    if seasonality < np.shape(past_data)[1]:
        forecast_freq = seasonality
    else:
        # edge case: the seasonal freq is larger than the length of ts
        forecast_freq = 1

    ndim = np.ndim(past_data)
    if ndim == 2:
        y_t = past_data[:, :-forecast_freq]
        y_tm = past_data[:, forecast_freq:]
    elif ndim:
        y_t = past_data[:, :-forecast_freq, :]
        y_tm = past_data[:, forecast_freq:, :]
    else:
        raise GluonTSUserError(
            "Input data is {ndim}-dimensional but must be two-dimensional "
            "(univariate case) or three-dimensional (multivariate case)"
        )

    return np.abs(y_t - y_tm)


def msis_numerator(
    data: EvalData,
    alpha: float,
) -> np.ndarray:
    """Calculates the numerator of the Mean Scaled Interval Score.

    Further calculation to get the true metric value happens in `MSIS`.
    """
    lower_quantile = data[str(alpha / 2)]
    upper_quantile = data[str(1.0 - alpha / 2)]
    label = data["label"]

    numerator = (
        upper_quantile
        - lower_quantile
        + 2.0 / alpha * (lower_quantile - label) * (label < lower_quantile)
        + 2.0 / alpha * (label - upper_quantile) * (label > upper_quantile)
    )

    return numerator
