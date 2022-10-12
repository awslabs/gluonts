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

from typing import Optional, Collection, Union
import numpy as np
from dataclasses import dataclass, field
from gluonts.ev_v5.api import Concat, Mean, SimpleMetric, Sum

from gluonts.exceptions import GluonTSUserError
from gluonts.time_feature import get_seasonality


# METRIC FUNCTIONS (non-aggregating)


def abs_label(data: dict):
    return np.abs(data["label"])


def error(data: dict, forecast_type: str = "mean"):
    return data["label"] - data[forecast_type]


def abs_error(data: dict, forecast_type: str = "mean"):
    return np.abs(error(data, forecast_type))


def squared_error(data: dict, forecast_type: str = "mean"):
    return np.square(error(data, forecast_type))


def quantile_loss(data: dict, q: float = 0.5):
    forecast_type = str(q)
    prediction = data[forecast_type]

    return np.abs(
        error(data, forecast_type) * ((prediction >= data["label"]) - q)
    )


def coverage(data: dict, q: float = 0.5):
    forecast_type = str(q)
    return data["label"] < data[forecast_type]


# METRICS USED IN EVALUATION

# SIMPLE METRICS

class AbsLabel(SimpleMetric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.metric_fn = abs_label
        self.super_aggregate = Concat()


class AbsLabelMean(SimpleMetric):
    def __init__(self, axis, **kwargs) -> None:
        super().__init__(**kwargs)
        self.metric_fn = abs_label
        self.super_aggregate = Mean(axis=axis)


class MSE(SimpleMetric):
    def __init__(self, axis: Optional[int] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.metric_fn = squared_error
        self.super_aggregate = Mean(axis=axis)


class QuantileLossSum(SimpleMetric):
    def __init__(self, axis: Optional[int] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.metric_fn = quantile_loss
        self.super_aggregate = Sum(axis=axis)

# DERIVED METRICS

class RMSE:
    def __init__(self, axis, **kwargs) -> None:
        self.mse = MSE(axis=axis, **kwargs)

    def step(self, data):
        self.mse.step(data)

    def get(self):
        return np.sqrt(self.mse.get())

class NRMSE:
    def __init__(self, axis:Optional[int], **kwargs) -> None:
        # TODO: use actual kwargs here 
        self.rmse = RMSE(axis=axis)
        self.abs_label_mean = AbsLabelMean(axis=axis)

    def step(self, data):
        self.rmse.step(data)
        self.abs_label_mean.step(data)

    def get(self):
        return self.rmse.get() / self.abs_label_mean.get()