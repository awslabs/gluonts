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

from typing import Dict

import numpy as np


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

    return 2 * np.abs(
        error(data, forecast_type) * ((prediction >= data["label"]) - q)
    )


def coverage(data: Dict[str, np.ndarray], q: float) -> np.ndarray:
    forecast_type = str(q)
    return (data["label"] <= data[forecast_type]).astype(float)


def absolute_percentage_error(
    data: Dict[str, np.ndarray], forecast_type: str
) -> np.ndarray:
    return absolute_error(data, forecast_type) / absolute_label(data)


def symmetric_absolute_percentage_error(
    data: Dict[str, np.ndarray], forecast_type: str
) -> np.ndarray:
    return (
        2
        * absolute_error(data, forecast_type)
        / (absolute_label(data) + np.abs(data[forecast_type]))
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


def absolute_scaled_error(
    data: Dict[str, np.ndarray], forecast_type: str
) -> np.ndarray:
    return absolute_error(data, forecast_type) / data["seasonal_error"]
