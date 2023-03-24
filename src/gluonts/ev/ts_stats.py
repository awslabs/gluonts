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


def seasonal_error(
    time_series: np.ndarray, seasonality: int, time_axis=0
) -> np.ndarray:
    """The mean abs. difference of a time series, shifted by its seasonality.

    Some metrics use the seasonal error for normalization."""

    time_length = time_series.shape[time_axis]

    if seasonality > time_length:
        seasonality = 1

    y_t = np.take(time_series, range(seasonality, time_length), axis=time_axis)
    y_tm = np.take(
        time_series, range(time_length - seasonality), axis=time_axis
    )

    return np.abs(y_t - y_tm).mean(axis=time_axis, keepdims=True)
