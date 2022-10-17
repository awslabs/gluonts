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

from dataclasses import dataclass
from functools import partial
from typing import Optional

from torch import absolute
import numpy as np
from gluonts.ev.api import DerivedMetric, Metric, MetricEvaluator, SimpleMetric, SimpleMetricEvaluator
from gluonts.ev.batch_aggregations import Mean, Sum
from gluonts.ev.metric_functions import (
    abs_error,
    abs_label,
    absolute_percentage_error,
    coverage,
    quantile_loss,
    squared_error,
    symmetric_absolute_percentage_error,
)


class Metric:
    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        raise NotImplementedError


@dataclass
class AbsLabelMean(Metric):
    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return SimpleMetricEvaluator(
            map=abs_label,
            aggregate=Mean(axis=axis),
        )


@dataclass
class AbsLabelSum(Metric):
    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return SimpleMetricEvaluator(
            map=abs_label,
            aggregate=Sum(axis=axis),
        )


@dataclass
class AbsErrorSum(Metric):
    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return SimpleMetricEvaluator(
            map=partial(abs_error, forecast_type=self.forecast_type),
            aggregate=Sum(axis=axis),
        )


@dataclass
class MSE(Metric):
    forecast_type: str = "mean"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return SimpleMetricEvaluator(
            map=partial(squared_error, forecast_type=self.forecast_type),
            aggregate=Mean(axis=axis),
        )


@dataclass
class QuantileLossSum(Metric):
    q: float = 0.5

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return SimpleMetricEvaluator(
            map=partial(quantile_loss, q=self.q),
            aggregate=Sum(axis=axis),
        )


# TODO: maybe just call this Coverage?
@dataclass
class CoverageMean(Metric):
    q: float = 0.5

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return SimpleMetricEvaluator(
            map=partial(coverage, q=self.q),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MAPE(Metric):
    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return SimpleMetricEvaluator(
            map=partial(absolute_percentage_error, q=self.q),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MAPE(Metric):
    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return SimpleMetricEvaluator(
            map=partial(symmetric_absolute_percentage_error, q=self.q),
            aggregate=Mean(axis=axis),
        )


@dataclass
class RMSE(Metric):
    forecast_type: str = "mean"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        def post_process(mse: np.ndarray) -> np.ndarray:
            return np.sqrt(mse)

        return DerivedMetric(
            metrics={"mse": MSE(forecast_type=self.forecast_type)(axis=axis)},
            post_process=post_process,
        )


@dataclass
class NRMSE(Metric):
    forecast_type: str = "mean"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        def post_process(
            rmse: np.ndarray, abs_label_mean: np.ndarray
        ) -> np.ndarray:
            return rmse / abs_label_mean

        return DerivedMetric(
            metrics={
                "rmse": RMSE()(axis=axis),
                "abs_label_mean": AbsLabelMean()(axis=axis),
            },
            post_process=post_process,
        )
