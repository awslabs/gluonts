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

# pylint: disable=missing-function-docstring
import math
import numpy as np
import numpy.typing as npt
from numpy import ma
from .quantile import QuantileForecasts


def rmse(
    y_pred: npt.NDArray[np.float32], y_true: npt.NDArray[np.float32]
) -> float:
    return np.sqrt(((y_pred - y_true) ** 2).mean())


def abs_error_sum(
    y_pred: npt.NDArray[np.float32], y_true: npt.NDArray[np.float32]
) -> float:
    return np.abs(y_pred - y_true).sum()


def abs_target_sum(y_true: npt.NDArray[np.float32]) -> float:
    return np.sum(np.abs(y_true))


def abs_target_mean(y_true: npt.NDArray[np.float32]) -> float:
    return np.mean(np.abs(y_true))


def naive_error(
    y_past: ma.masked_array,  # type: ignore
    seasonality: int,
) -> npt.NDArray[np.float32]:
    error = np.abs(y_past[:, seasonality:] - y_past[:, :-seasonality]).mean(1)
    return ma.getdata(error)


def mase(
    y_pred: npt.NDArray[np.float32],
    y_true: npt.NDArray[np.float32],
    error: npt.NDArray[np.float32],
) -> float:
    mase_values = np.abs(y_pred - y_true).mean(1) / error
    return mase_values.mean()


def smape(
    y_pred: npt.NDArray[np.float32], y_true: npt.NDArray[np.float32]
) -> float:
    median = y_pred
    num = np.abs(y_true - median)
    denom = (np.abs(y_true) + np.abs(median)) / 2
    # If the denominator is 0, we set it to float('inf') such that any division yields 0 (this
    # might not be fully mathematically correct, but at least we don't get NaNs)
    denom[denom == 0] = math.inf
    return np.mean(num / denom, axis=1).mean()


def ncrps(y_pred: QuantileForecasts, y_true: npt.NDArray[np.float32]) -> float:
    y_true_rep = y_true[:, None].repeat(len(y_pred.quantiles), axis=1)
    quantiles = np.array([float(q) for q in y_pred.quantiles])
    quantile_losses = 2 * np.sum(
        np.abs(
            (y_pred.values - y_true_rep)
            * ((y_true_rep <= y_pred.values) - quantiles[:, None])
        ),
        axis=-1,
    )  # shape [num_time_series, num_quantiles]
    denom = np.sum(np.abs(y_true))  # shape [1]
    weighted_losses = quantile_losses.sum(0) / denom  # shape [num_quantiles]
    return weighted_losses.mean()
