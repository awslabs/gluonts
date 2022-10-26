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

from .evaluator import (
    DerivedEvaluator,
    DirectEvaluator,
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


def mean_absolute_label(
    axis: Optional[int] = None, name: str = "mean_absolute_label"
) -> DirectEvaluator:
    return DirectEvaluator(
        map=absolute_label, aggregate=Mean(axis=axis), name=name
    )


def sum_absolute_label(
    axis: Optional[int] = None, name: str = "sum_absolute_label"
) -> DirectEvaluator:
    return DirectEvaluator(
        name=name, map=absolute_label, aggregate=Sum(axis=axis)
    )


@dataclass
class SumAbsoluteError:
    forecast_type: str = "0.5"

    def __call__(
        self, axis: Optional[int] = None, name: str = "sum_absolute_error"
    ) -> DirectEvaluator:
        return DirectEvaluator(
            name=name,
            map=partial(absolute_error, forecast_type=self.forecast_type),
            aggregate=Sum(axis=axis),
        )


@dataclass
class MSE:
    """Mean Squared Error"""

    forecast_type: str = "mean"

    def __call__(
        self, axis: Optional[int] = None, name: str = "MSE"
    ) -> DirectEvaluator:
        return DirectEvaluator(
            name=name,
            map=partial(squared_error, forecast_type=self.forecast_type),
            aggregate=Mean(axis=axis),
        )


@dataclass
class SumQuantileLoss:
    q: float

    def __call__(
        self, axis: Optional[int] = None, name: str = "sum_quantile_loss"
    ) -> DirectEvaluator:
        return DirectEvaluator(
            name=name,
            map=partial(quantile_loss, q=self.q),
            aggregate=Sum(axis=axis),
        )


@dataclass
class Coverage:
    q: float

    def __call__(
        self, axis: Optional[int] = None, name: str = "coverage"
    ) -> DirectEvaluator:
        return DirectEvaluator(
            name=name,
            map=partial(coverage, q=self.q),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MAPE:
    """Mean Absolute Percentage Error"""

    forecast_type: str = "0.5"

    def __call__(
        self, axis: Optional[int] = None, name: str = "MAPE"
    ) -> DirectEvaluator:
        return DirectEvaluator(
            name=name,
            map=partial(
                absolute_percentage_error, forecast_type=self.forecast_type
            ),
            aggregate=Mean(axis=axis),
        )


@dataclass
class SMAPE:
    """Symmetric Mean Absolute Percentage Error"""

    forecast_type: str = "0.5"

    def __call__(
        self, axis: Optional[int] = None, name: str = "sMAPE"
    ) -> DirectEvaluator:
        return DirectEvaluator(
            name=name,
            map=partial(
                symmetric_absolute_percentage_error,
                forecast_type=self.forecast_type,
            ),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MSIS:
    """Mean Scaled Interval Score"""

    alpha: float = 0.05

    def __call__(
        self, axis: Optional[int] = None, name: str = "MSIS"
    ) -> DirectEvaluator:
        return DirectEvaluator(
            name=name,
            map=partial(scaled_interval_score, alpha=self.alpha),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MASE:
    """Mean Absolute Scaled Error"""

    forecast_type: str = "0.5"

    def __call__(
        self, axis: Optional[int] = None, name: str = "MASE"
    ) -> DirectEvaluator:
        return DirectEvaluator(
            name=name,
            map=partial(
                absolute_scaled_error, forecast_type=self.forecast_type
            ),
            aggregate=Mean(axis=axis),
        )


@dataclass
class ND:
    """Normalized Deviation"""

    forecast_type: str = "mean"

    @staticmethod
    def normalized_deviation(
        sum_absolute_error: np.ndarray, sum_absolute_label: np.ndarray
    ) -> np.ndarray:
        return sum_absolute_error / sum_absolute_label

    def __call__(
        self, axis: Optional[int] = None, name="ND"
    ) -> DerivedEvaluator:
        return DerivedEvaluator(
            name=name,
            evaluators={
                "sum_absolute_error": SumAbsoluteError(
                    forecast_type=self.forecast_type
                )(axis=axis),
                "sum_absolute_label": sum_absolute_label(axis=axis),
            },
            post_process=self.normalized_deviation,
        )


@dataclass
class RMSE:
    """Root Mean Squared Error"""

    forecast_type: str = "mean"

    @staticmethod
    def root_mean_squared_error(mean_squared_error: np.ndarray) -> np.ndarray:
        return np.sqrt(mean_squared_error)

    def __call__(
        self, axis: Optional[int] = None, name: str = "RMSE"
    ) -> DerivedEvaluator:
        return DerivedEvaluator(
            name=name,
            evaluators={
                "mean_squared_error": MSE(forecast_type=self.forecast_type)(
                    axis=axis
                )
            },
            post_process=self.root_mean_squared_error,
        )


@dataclass
class NRMSE:
    """Normalized Root Mean Squared Error"""

    forecast_type: str = "mean"

    @staticmethod
    def normalize_root_mean_squared_error(
        root_mean_squared_error: np.ndarray, mean_absolute_label: np.ndarray
    ) -> np.ndarray:
        return root_mean_squared_error / mean_absolute_label

    def __call__(
        self, axis: Optional[int] = None, name: str = "NRMSE"
    ) -> DerivedEvaluator:
        return DerivedEvaluator(
            name=name,
            evaluators={
                "root_mean_squared_error": RMSE(
                    forecast_type=self.forecast_type
                )(axis=axis),
                "mean_absolute_label": mean_absolute_label(axis=axis),
            },
            post_process=self.normalize_root_mean_squared_error,
        )


@dataclass
class WeightedSumQuantileLoss:
    q: float

    @staticmethod
    def weight_sum_quantile_loss(
        sum_quantile_loss: np.ndarray, sum_absolute_label: np.ndarray
    ) -> np.ndarray:
        return sum_quantile_loss / sum_absolute_label

    def __call__(
        self,
        axis: Optional[int] = None,
        name: str = "weighted_sum_quantile_loss",
    ) -> DerivedEvaluator:
        return DerivedEvaluator(
            name=name,
            evaluators={
                "sum_quantile_loss": SumQuantileLoss(q=self.q)(axis=axis),
                "sum_absolute_label": sum_absolute_label(axis=axis),
            },
            post_process=self.weight_sum_quantile_loss,
        )
