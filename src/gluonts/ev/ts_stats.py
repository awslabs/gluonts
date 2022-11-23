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


def seasonal_error(time_series: np.ndarray, seasonality: int) -> np.ndarray:
    """The mean abs. difference of a time series, shifted by its seasonality.

    Some metrics use the seasonal error for normalization."""

    if seasonality < time_series.shape[-1]:
        forecast_freq = seasonality
    else:
        # edge case: the seasonal freq is larger than the length of ts
        forecast_freq = 1

    if time_series.ndim == 1:
        y_t = time_series[:-forecast_freq]
        y_tm = time_series[forecast_freq:]

        return np.abs(y_t - y_tm).mean(keepdims=True)
    else:
        # multivariate case:
        # time_series has shape (# of time stamps, # of variates)
        y_t = time_series[:-forecast_freq, :]
        y_tm = time_series[forecast_freq:, :]

        return np.abs(y_t - y_tm).mean(axis=0, keepdims=True)
