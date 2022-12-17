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

from typing import Optional

import numpy as np

from gluonts.time_feature import get_seasonality


def calculate_seasonal_error(
    past_data: np.ndarray,
    freq: Optional[str] = None,
    seasonality: Optional[int] = None,
):
    r"""
    .. math::

        seasonal\_error = mean(|Y[t] - Y[t-m]|)

    where m is the seasonal frequency. See [HA21]_ for more details.
    """
    # Check if the length of the time series is larger than the seasonal
    # frequency
    if not seasonality:
        assert freq is not None, "Either freq or seasonality must be provided"
        seasonality = get_seasonality(freq)

    if seasonality < len(past_data):
        forecast_freq = seasonality
    else:
        # edge case: the seasonal freq is larger than the length of ts
        # revert to freq=1

        # logging.info('The seasonal frequency is larger than the length of the
        # time series. Reverting to freq=1.')
        forecast_freq = 1

    y_t = past_data[:-forecast_freq]
    y_tm = past_data[forecast_freq:]

    return np.mean(abs(y_t - y_tm))


def mse(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        mse = mean((Y - \hat{Y})^2)

    See [HA21]_ for more details.
    """
    return np.mean(np.square(target - forecast))


def abs_error(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        abs\_error = sum(|Y - \hat{Y}|)
    """
    return np.sum(np.abs(target - forecast))


def quantile_loss(target: np.ndarray, forecast: np.ndarray, q: float) -> float:
    r"""
    .. math::

        quantile\_loss = 2 * sum(|(Y - \hat{Y}) * (Y <= \hat{Y}) - q|)
    """
    return 2 * np.sum(np.abs((forecast - target) * ((target <= forecast) - q)))


def coverage(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        coverage = mean(Y < \hat{Y})
    """
    return np.mean(target < forecast)


def mase(
    target: np.ndarray,
    forecast: np.ndarray,
    seasonal_error: float,
) -> float:
    r"""
    .. math::

        mase = mean(|Y - \hat{Y}|) / seasonal\_error

    See [HA21]_ for more details.
    """
    return np.mean(np.abs(target - forecast)) / seasonal_error


def mape(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        mape = mean(|Y - \hat{Y}| / |Y|))

    See [HA21]_ for more details.
    """
    return np.mean(np.abs(target - forecast) / np.abs(target))


def smape(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        smape = 2 * mean(|Y - \hat{Y}| / (|Y| + |\hat{Y}|))

    See [HA21]_ for more details.
    """
    return 2 * np.mean(
        np.abs(target - forecast) / (np.abs(target) + np.abs(forecast))
    )


def msis(
    target: np.ndarray,
    lower_quantile: np.ndarray,
    upper_quantile: np.ndarray,
    seasonal_error: float,
    alpha: float,
) -> float:
    r"""
    .. math::

        msis = mean(U - L + 2/alpha * (L-Y) * I[Y<L] + 2/alpha * (Y-U) * I[Y>U]) / seasonal\_error

    See [SSA20]_ for more details.
    """  # noqa: E501

    numerator = np.mean(
        upper_quantile
        - lower_quantile
        + 2.0 / alpha * (lower_quantile - target) * (target < lower_quantile)
        + 2.0 / alpha * (target - upper_quantile) * (target > upper_quantile)
    )

    return numerator / seasonal_error


def abs_target_sum(target) -> float:
    r"""
    .. math::

        abs\_target\_sum = sum(|Y|)
    """
    return np.sum(np.abs(target))


def abs_target_mean(target) -> float:
    r"""
    .. math::

        abs\_target\_mean = mean(|Y|)
    """
    return np.mean(np.abs(target))
