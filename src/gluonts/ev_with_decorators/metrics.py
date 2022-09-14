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
from typing import Optional, Union, Dict

import numpy as np

from gluonts.ev_with_decorators.api import (
    Input,
    metric,
    aggregate,
    BaseMetric,
    AggregateMetric,
    ForecastBatch,
)
from gluonts.time_feature import get_seasonality


def standardize_error_type(error_type: Union[float, str]) -> Union[float, str]:
    # this function returns either a float to be interpreted as a quantile or "mean"
    if error_type == "mean":
        return "mean"
    if error_type == "median":
        return 0.5
    return float(error_type)


# DYNAMIC OBJECTS
class Error(BaseMetric):
    def __init__(self):
        self.name = "error"
        self.dependencies = (
            Input("target"),
            Input("forecast_batch"),
        )
        self.target = None
        self.forecast_batch = None
        self.cache: Dict[Union[float, str], np.ndarray] = dict()

    def apply(self, data: dict):
        # instead of writing hard data, we get the non-changing target for later use
        self.target = data["target"]
        self.forecast_batch = data["forecast_batch"]

        data[self.name] = self  # make error available to dependents

    def __getitem__(self, error_type: Union[float, str]):
        error_type = standardize_error_type(error_type)

        if error_type not in self.cache:
            if error_type == "mean":
                self.cache[error_type] = self.target - self.forecast_batch.mean
            else:
                self.cache[
                    error_type
                ] = self.target - self.forecast_batch.quantile(
                    float(error_type)
                )

        return self.cache[error_type]


class AbsError(BaseMetric):
    def __init__(self):
        self.name = "abs_error"
        self.dependencies = (Error(),)
        self.error = None
        self.cache: Dict[Union[float, str], np.ndarray] = dict()

    def apply(self, data: dict):
        self.error = data["error"]
        data[self.name] = self  # make abs_error available to dependents

    def __getitem__(self, error_type: Union[float, str]):
        error_type = standardize_error_type(error_type)

        if error_type not in self.cache:
            self.cache[error_type] = np.abs(self.error[error_type])

        return self.cache[error_type]


class SquaredError(BaseMetric):
    def __init__(self):
        self.name = "squared_error"
        self.dependencies = (Error(),)
        self.error = None
        self.cache: Dict[Union[float, str], np.ndarray] = dict()

    def apply(self, data: dict):
        self.error = data["error"]
        data[self.name] = self  # make squared_error available to dependents

    def __getitem__(self, error_type: Union[float, str]):
        error_type = standardize_error_type(error_type)

        if error_type not in self.cache:
            self.cache[error_type] = np.square(self.error[error_type])

        return self.cache[error_type]


# BASE METRICS (two-dimensional in univariate case)
@metric(Input("target"))
def abs_target(target):
    return np.abs(target)


# AGGREGATIONS (have lower number of dimensions than inputs and base metrics)
@aggregate(SquaredError())
def mse(squared_error: SquaredError, axis: Optional[int] = None):
    return np.mean(squared_error["mean"], axis=axis)


@aggregate(mse)
def rmse(mse: np.ndarray, axis: Optional[int] = None):
    return np.sqrt(mse)  # axis is already considered


@aggregate(rmse, abs_target)
def nrmse(
    rmse: np.ndarray, abs_target: np.ndarray, axis: Optional[int] = None
):
    return rmse / np.mean(abs_target, axis=axis)


@aggregate(AbsError(), abs_target)
def mape(
    abs_error: AbsError,
    abs_target: np.ndarray,
    axis: Optional[int] = None,
):
    return np.mean(abs_error["median"] / abs_target, axis=axis)


@aggregate(AbsError(), abs_target, Input("forecast_batch"))
def smape(
    abs_error: AbsError,
    abs_target: np.ndarray,
    forecast_batch: ForecastBatch,
    axis: Optional[int] = None,
):
    return 2 * np.mean(
        abs_error["median"] / (abs_target + np.abs(forecast_batch.median)),
        axis=axis,
    )


@aggregate(AbsError(), abs_target)
def nd(
    abs_error: AbsError,
    abs_target: np.ndarray,
    axis: int,
):
    return np.sum(abs_error["median"], axis=axis) / np.sum(
        abs_target, axis=axis
    )


# for metrics with extra parameters, we declare classes directly instead of using decorators
class QuantileLoss(BaseMetric):
    def __init__(self, q):
        self.name = f"quantile_loss[{q}]"
        self.dependencies = (
            Input("target"),
            Input(f"forecast_batch"),
        )
        self.q = q

    def fn(self, target: np.ndarray, forecast_batch: ForecastBatch):
        prediction_q = forecast_batch.quantile(self.q)
        return np.abs(
            (target - prediction_q) * ((prediction_q >= target) - self.q)
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
