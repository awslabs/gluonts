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


def absolute_percentage_error(data: EvalData, forecast_type: str) -> np.ndarray:
    return abs_error(data, forecast_type) / abs_label(data)


def symmetric_absolute_percentage_error(
    data: EvalData, forecast_type: str
) -> np.ndarray:
    return abs_error(data, forecast_type) / (
        abs_label(data) + np.abs(data[forecast_type])
    )
