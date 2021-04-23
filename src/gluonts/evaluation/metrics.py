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
import pandas as pd
from gluonts.model.forecast import Forecast
from gluonts.time_feature import get_seasonality


def calculate_seasonal_error(
    past_data: np.ndarray,
    forecast: Forecast,
    seasonality: Optional[int] = None,
):
    r"""
    .. math::

        seasonal_error = mean(|Y[t] - Y[t-m]|)

    where m is the seasonal frequency
    https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
    """
    # Check if the length of the time series is larger than the seasonal frequency
    if not seasonality:
        seasonality = get_seasonality(forecast.freq)

    if seasonality < len(past_data):
        forecast_freq = seasonality
    else:
        # edge case: the seasonal freq is larger than the length of ts
        # revert to freq=1
        # logging.info('The seasonal frequency is larger than the length of the time series. Reverting to freq=1.')
        forecast_freq = 1

    y_t = past_data[:-forecast_freq]
    y_tm = past_data[forecast_freq:]

    return np.mean(abs(y_t - y_tm))


def mse(target: np.ndarray, forecast: np.ndarray) -> float:
    return np.mean(np.square(target - forecast))


def abs_error(target: np.ndarray, forecast: np.ndarray) -> float:
    return np.sum(np.abs(target - forecast))


def quantile_loss(target: np.ndarray, forecast: np.ndarray, q: float) -> float:
    return 2 * np.sum(np.abs((forecast - target) * ((target <= forecast) - q)))


def coverage(target: np.ndarray, forecast: np.ndarray) -> float:
    return np.mean(target < forecast)


def mase(
    target: np.ndarray,
    forecast: np.ndarray,
    seasonal_error: float,
) -> float:
    r"""
    .. math::

        mase = mean(|Y - Y_hat|) / seasonal_error

    https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
    """
    return np.mean(np.abs(target - forecast)) / seasonal_error


def mape(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        mape = mean(|Y - Y_hat| / |Y|))
    """
    return np.mean(np.abs(target - forecast) / np.abs(target))


def smape(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        smape = 2 * mean(|Y - Y_hat| / (|Y| + |Y_hat|))

    https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
    """
    return 2 * np.mean(
        np.abs(target - forecast) / (np.abs(target) + np.abs(forecast))
    )


def owa(
    target: np.ndarray,
    forecast: np.ndarray,
    past_data: np.ndarray,
    seasonal_error: float,
    start_date: pd.Timestamp,
) -> float:
    r"""
    .. math::

        owa = 0.5*(smape/smape_naive + mase/mase_naive)

    https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
    """
    # avoid import error due to circular dependency
    from gluonts.model.naive_2 import naive_2

    # calculate the forecast of the seasonal naive predictor
    naive_median_fcst = naive_2(
        past_data, len(target), freq=start_date.freqstr
    )

    return 0.5 * (
        (smape(target, forecast) / smape(target, naive_median_fcst))
        + (
            mase(target, forecast, seasonal_error)
            / mase(target, naive_median_fcst, seasonal_error)
        )
    )


def msis(
    target: np.ndarray,
    lower_quantile: np.ndarray,
    upper_quantile: np.ndarray,
    seasonal_error: float,
    alpha: float,
) -> float:
    r"""
    :math:

        msis = mean(U - L + 2/alpha * (L-Y) * I[Y<L] + 2/alpha * (Y-U) * I[Y>U]) / seasonal_error

    https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
    """
    numerator = np.mean(
        upper_quantile
        - lower_quantile
        + 2.0 / alpha * (lower_quantile - target) * (target < lower_quantile)
        + 2.0 / alpha * (target - upper_quantile) * (target > upper_quantile)
    )

    return numerator / seasonal_error


def abs_target_sum(target) -> float:
    return np.sum(np.abs(target))


def abs_target_mean(target) -> float:
    return np.mean(np.abs(target))
