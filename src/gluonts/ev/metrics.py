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

import numpy as np

from gluonts.exceptions import GluonTSUserError
from .api import (
    DerivedMetricEvaluator,
    Metric,
    MetricEvaluator,
    StandardMetricEvaluator,
)
from .aggregations import Mean, Sum
from .stats import (
    absolute_error,
    absolute_label,
    absolute_percentage_error,
    absolute_scaled_error,
    coverage,
    quantile_loss,
    scaled_interval_score,
    squared_error,
    symmetric_absolute_percentage_error,
)


@dataclass
class AbsoluteLabelMean(Metric):
    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return StandardMetricEvaluator(
            map=absolute_label,
            aggregate=Mean(axis=axis),
        )


@dataclass
class AbsoluteLabelSum(Metric):
    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return StandardMetricEvaluator(
            map=absolute_label,
            aggregate=Sum(axis=axis),
        )


@dataclass
class AbsoluteErrorSum(Metric):
    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return StandardMetricEvaluator(
            map=partial(absolute_error, forecast_type=self.forecast_type),
            aggregate=Sum(axis=axis),
        )


@dataclass
class MeanSquaredError(Metric):
    forecast_type: str = "mean"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return StandardMetricEvaluator(
            map=partial(squared_error, forecast_type=self.forecast_type),
            aggregate=Mean(axis=axis),
        )


@dataclass
class QuantileLoss(Metric):
    q: float = 0.5

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return StandardMetricEvaluator(
            map=partial(quantile_loss, q=self.q),
            aggregate=Sum(axis=axis),
        )


@dataclass
class Coverage(Metric):
    q: float = 0.5

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return StandardMetricEvaluator(
            map=partial(coverage, q=self.q),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MeanAbsolutePercentageError(Metric):
    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return StandardMetricEvaluator(
            map=partial(
                absolute_percentage_error, forecast_type=self.forecast_type
            ),
            aggregate=Mean(axis=axis),
        )


@dataclass
class SymmetricMeanAbsolutePercentageError(Metric):
    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return StandardMetricEvaluator(
            map=partial(
                symmetric_absolute_percentage_error,
                forecast_type=self.forecast_type,
            ),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MeanScaledIntervalScore(Metric):
    alpha: float = 0.05

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        return StandardMetricEvaluator(
            map=partial(scaled_interval_score, alpha=self.alpha),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MeanAbsoluteScaledError(Metric):
    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        if axis not in [None, 1]:
            raise GluonTSUserError(
                f"MASE requires 'axis' to be None or 1 (not {axis})"
            )

        return StandardMetricEvaluator(
            map=partial(
                absolute_scaled_error, forecast_type=self.forecast_type
            ),
            aggregate=Mean(axis=axis),
        )


@dataclass
class NormalizedDeviation(Metric):
    forecast_type: str = "mean"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        def post_process(
            abs_error_sum: np.ndarray, abs_label_sum: np.ndarray
        ) -> np.ndarray:
            return abs_error_sum / abs_label_sum

        return DerivedMetricEvaluator(
            metrics={
                "abs_error_sum": AbsoluteErrorSum(
                    forecast_type=self.forecast_type
                )(axis=axis),
                "abs_label_sum": AbsoluteLabelSum()(axis=axis),
            },
            post_process=post_process,
        )


@dataclass
class RootMeanSquaredError(Metric):
    forecast_type: str = "mean"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        def post_process(mse: np.ndarray) -> np.ndarray:
            return np.sqrt(mse)

        return DerivedMetricEvaluator(
            metrics={
                "mse": MeanSquaredError(forecast_type=self.forecast_type)(
                    axis=axis
                )
            },
            post_process=post_process,
        )


@dataclass
class NormalizedRootMeanSquaredError(Metric):
    forecast_type: str = "mean"

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        def post_process(
            rmse: np.ndarray, abs_label_mean: np.ndarray
        ) -> np.ndarray:
            return rmse / abs_label_mean

        return DerivedMetricEvaluator(
            metrics={
                "rmse": RootMeanSquaredError()(axis=axis),
                "abs_label_mean": AbsoluteLabelMean()(axis=axis),
            },
            post_process=post_process,
        )


@dataclass
class WeightedQuantileLoss:
    q: float = 0.5

    def __call__(self, axis: Optional[int] = None) -> MetricEvaluator:
        def post_process(
            quantile_loss: np.ndarray, abs_label_sum: np.ndarray
        ) -> np.ndarray:
            return quantile_loss / abs_label_sum

        return DerivedMetricEvaluator(
            metrics={
                "quantile_loss": QuantileLoss()(axis=axis),
                "abs_label_sum": AbsoluteLabelSum()(axis=axis),
            },
            post_process=post_process,
        )
