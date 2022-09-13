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

from gluonts.ev_with_decorators.api import (
    Input,
    metric,
    aggregate,
    BaseMetric,
    AggregateMetric,
)

# BASE METRICS (two-dimensional in univariate case)
from gluonts.time_feature import get_seasonality


@metric(Input("target"))
def abs_target(target):
    return np.abs(target)


@metric(Input("prediction"))
def abs_prediction_mean(prediction_mean):
    return np.abs(prediction_mean)


@metric(Input("prediction_median"))
def abs_prediction_median(prediction_median):
    return np.abs(prediction_median)


@metric(Input("target"), Input("prediction_mean"))
def error_wrt_mean(target, prediction_mean):
    return target - prediction_mean


@metric(Input("target"), Input("prediction_median"))
def error_wrt_median(target, prediction_median):
    return target - prediction_median


@metric(error_wrt_median)
def abs_error_wrt_median(error_wrt_median):
    return np.abs(error_wrt_median)


@metric(error_wrt_mean)
def squared_error_wrt_mean(error_wrt_mean):
    return error_wrt_mean * error_wrt_mean


# AGGREGATIONS (have lower number of dimensions than inputs and base metrics)
@aggregate(abs_target)
def abs_target_sum(abs_target: np.ndarray, axis: Optional[int] = None):
    return np.sum(abs_target, axis=axis)


@aggregate(abs_target)
def abs_target_mean(abs_target: np.ndarray, axis: Optional[int] = None):
    return np.mean(abs_target, axis=axis)


@aggregate(squared_error_wrt_mean)
def mse(squared_error_wrt_mean: np.ndarray, axis: Optional[int] = None):
    return np.mean(squared_error_wrt_mean, axis=axis)


@aggregate(mse)
def rmse(mse: np.ndarray, axis: Optional[int] = None):
    return np.sqrt(mse)  # axis is already considered


@aggregate(rmse, abs_target_mean)
def nrmse(
    rmse: np.ndarray, abs_target_mean: np.ndarray, axis: Optional[int] = None
):
    return rmse / abs_target_mean  # axis is already considered


@aggregate(abs_error_wrt_median, abs_target)
def mape(
    abs_error_wrt_median: np.ndarray,
    abs_target: np.ndarray,
    axis: Optional[int] = None,
):
    return np.mean(abs_error_wrt_median / abs_target, axis=axis)


@aggregate(abs_error_wrt_median, abs_target, abs_prediction_median)
def smape(
    abs_error_wrt_median: np.ndarray,
    abs_target: np.ndarray,
    abs_prediction_median: np.ndarray,
    axis: Optional[int] = None,
):
    return 2 * np.mean(
        abs_error_wrt_median / (abs_target + abs_prediction_median), axis=axis
    )


@aggregate(abs_error_wrt_median)
def abs_error_wrt_median_sum(abs_error_wrt_median: np.ndarray, axis: int):
    return np.sum(abs_error_wrt_median, axis=axis)


@aggregate(abs_error_wrt_median_sum, abs_target_sum)
def nd(
    abs_error_wrt_median_sum: np.ndarray,
    abs_target_sum: np.ndarray,
    axis: int,
):
    return (
        abs_error_wrt_median_sum / abs_target_sum
    )  # axis is already considered


# for metrics with extra parameters, we declare classes directly instead of using decorators
class QuantileLoss(BaseMetric):
    def __init__(self, q):
        self.name = f"quantile_loss[{q}]"
        self.dependencies = (
            Input("target"),
            Input(f"prediction_quantile[{q}]"),
        )
        self.q = q

    def fn(self, target: np.ndarray, prediction_q: np.ndarray):
        return np.abs(
            (target - prediction_q) * ((prediction_q >= target) - self.q)
        )

    def apply(self, data):
        data[self.name] = self.fn(
            target=data["target"],
            prediction_q=data[f"prediction_quantile[{self.q}]"],
        )


class Coverage(BaseMetric):
    pass  # TODO


class SeasonalError(AggregateMetric):
    def __init__(
        self, freq: Optional[str] = None, seasonality: Optional[int] = None
    ):
        self.name = "seasonal_error"
        self.dependencies = (Input("past_data"),)
        self.freq = freq
        self.seasonality = seasonality

    def fn(self, past_data: np.ndarray, axis: Optional[int] = None):
        if not self.seasonality:
            assert (
                self.freq is not None
            ), "Either freq or seasonality must be provided"
            self.seasonality = get_seasonality(self.freq)

        if self.seasonality < len(past_data):
            forecast_freq = self.seasonality
        else:
            # edge case: the seasonal freq is larger than the length of ts
            forecast_freq = 1

        # TODO: using a dynamic axis gets ugly here - what can we do?
        if np.ndim(past_data) != 2:
            raise ValueError(
                "Seasonal error can't handle input data that is not 2-dimensional"
            )
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


class MASE(AggregateMetric):
    pass  # TODO


class MSIS(AggregateMetric):
    pass  # TODO


class WeightedQuantileLoss(AggregateMetric):
    pass  # TODO


class OWA(AggregateMetric):
    pass  # TODO
