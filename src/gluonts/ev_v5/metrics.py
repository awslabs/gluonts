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
from gluonts.ev_v5.api import Concat, Mean, Metric, SimpleMetric, Sum

from gluonts.time_feature import get_seasonality


# METRIC FUNCTIONS (these are non-aggregating and to be applied batch-wise)


def abs_label(data: dict):
    return np.abs(data["label"])


def error(data: dict, forecast_type: str):
    return data["label"] - data[forecast_type]


def abs_error(data: dict, forecast_type: str):
    return np.abs(error(data, forecast_type))


def squared_error(data: dict, forecast_type: str):
    return np.square(error(data, forecast_type))


def quantile_loss(data: dict, q: float):
    forecast_type = str(q)
    prediction = data[forecast_type]

    return np.abs(
        error(data, forecast_type) * ((prediction >= data["label"]) - q)
    )


def coverage(data: dict, q: float):
    forecast_type = str(q)
    return data["label"] < data[forecast_type]


def absolute_percentage_error(data: dict, forecast_type: str = "0.5"):
    return abs_error(data, forecast_type) / abs_label(data)


def symmetric_absolute_percentage_error(
    data: dict, forecast_type: str = "0.5"
):
    return abs_error(data, forecast_type) / (
        abs_label(data) + np.abs(data[forecast_type])
    )


# METRICS USED IN EVALUATION


class AbsLabel(SimpleMetric):
    def __init__(self) -> None:
        super().__init__()
        self.metric_fn = abs_label
        self.aggregate = Concat()


class Error(SimpleMetric):
    def __init__(self, forecast_type: str = "mean") -> None:
        super().__init__(forecast_type=forecast_type)
        self.metric_fn = error
        self.aggregate = Concat()


class AbsError(SimpleMetric):
    def __init__(self, forecast_type: str = "mean") -> None:
        super().__init__(forecast_type=forecast_type)
        self.metric_fn = abs_error
        self.aggregate = Concat()


class SquaredError(SimpleMetric):
    def __init__(self, forecast_type: str = "mean") -> None:
        super().__init__(forecast_type=forecast_type)
        self.metric_fn = squared_error
        self.aggregate = Concat()


class QuantileLoss(SimpleMetric):
    def __init__(self, q: float = 0.5) -> None:
        super().__init__(q=q)
        self.metric_fn = squared_error
        self.aggregate = Concat()


class Coverage(SimpleMetric):
    def __init__(self, q: float = 0.5) -> None:
        super().__init__(q=q)
        self.metric_fn = coverage
        self.aggregate = Concat()


class AbsLabelMean(SimpleMetric):
    def __init__(self, axis: Optional[int] = None) -> None:
        super().__init__()
        self.metric_fn = abs_label
        self.aggregate = Mean(axis=axis)


class AbsLabelSum(SimpleMetric):
    def __init__(self, axis: Optional[int] = None) -> None:
        super().__init__()
        self.metric_fn = abs_label
        self.aggregate = Sum(axis=axis)


class AbsErrorSum(SimpleMetric):
    def __init__(
        self, axis: Optional[int] = None, forecast_type: str = "0.5"
    ) -> None:
        super().__init__(forecast_type=forecast_type)
        self.metric_fn = abs_error
        self.aggregate = Sum(axis=axis)


class MSE(SimpleMetric):
    def __init__(
        self, axis: Optional[int] = None, forecast_type: str = "mean"
    ) -> None:
        super().__init__(forecast_type=forecast_type)
        self.metric_fn = squared_error
        self.aggregate = Mean(axis=axis)


class QuantileLossSum(SimpleMetric):
    def __init__(self, axis: Optional[int] = None, q: float = 0.5) -> None:
        super().__init__(q=q)
        self.metric_fn = quantile_loss
        self.aggregate = Sum(axis=axis)


# TODO: maybe just call this coverage?
class CoverageMean(SimpleMetric):
    def __init__(self, axis: Optional[int] = None, q: float = 0.5) -> None:
        super().__init__(q=q)
        self.metric_fn = coverage
        self.aggregate = Mean(axis=axis)


class MAPE(SimpleMetric):
    def __init__(
        self, axis: Optional[int] = None, forecast_type: str = "0.5"
    ) -> None:
        super().__init__(forecast_type=forecast_type)
        self.metric_fn = absolute_percentage_error
        self.aggregate = Mean(axis=axis)


class SMAPE(SimpleMetric):
    def __init__(
        self, axis: Optional[int] = None, forecast_type: str = "0.5"
    ) -> None:
        super().__init__(forecast_type=forecast_type)
        self.metric_fn = symmetric_absolute_percentage_error
        self.aggregate = Mean(axis=axis)


# DERIVED METRICS


class RMSE(Metric):
    def __init__(
        self, axis: Optional[int] = None, forecast_type: str = "mean"
    ) -> None:
        self.mse = MSE(axis=axis, forecast_type=forecast_type)

    def step(self, data):
        self.mse.step(data)

    def get(self):
        return np.sqrt(self.mse.get())


class NRMSE(Metric):
    def __init__(
        self, axis: Optional[int] = None, forecast_type: str = "mean"
    ) -> None:
        self.rmse = RMSE(axis=axis, forecast_type=forecast_type)
        self.abs_label_mean = AbsLabelMean(axis=axis)

    def step(self, data):
        self.rmse.step(data)
        self.abs_label_mean.step(data)

    def get(self):
        return self.rmse.get() / self.abs_label_mean.get()
