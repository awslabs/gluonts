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
from typing import Collection, Optional
from typing_extensions import Protocol, runtime_checkable

import numpy as np

from .evaluator import DirectEvaluator, DerivedEvaluator, Evaluator
from .aggregations import Mean, Sum
from .stats import (
    error,
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


@runtime_checkable
class Metric(Protocol):
    def __call__(self, axis: Optional[int] = None) -> Evaluator:
        raise NotImplementedError


def mean_absolute_label(axis: Optional[int] = None) -> DirectEvaluator:
    return DirectEvaluator(
        name="mean_absolute_label",
        stat=absolute_label,
        aggregate=Mean(axis=axis),
    )


def sum_absolute_label(axis: Optional[int] = None) -> DirectEvaluator:
    return DirectEvaluator(
        name="sum_absolute_label",
        stat=absolute_label,
        aggregate=Sum(axis=axis),
    )


@dataclass
class SumError:
    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> DirectEvaluator:
        return DirectEvaluator(
            name="sum_error",
            stat=partial(error, forecast_type=self.forecast_type),
            aggregate=Sum(axis=axis),
        )


@dataclass
class SumAbsoluteError:
    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> DirectEvaluator:
        return DirectEvaluator(
            name="sum_absolute_error",
            stat=partial(absolute_error, forecast_type=self.forecast_type),
            aggregate=Sum(axis=axis),
        )


@dataclass
class MSE:
    """Mean Squared Error"""

    forecast_type: str = "mean"

    def __call__(self, axis: Optional[int] = None) -> DirectEvaluator:
        return DirectEvaluator(
            name="MSE",
            stat=partial(squared_error, forecast_type=self.forecast_type),
            aggregate=Mean(axis=axis),
        )


@dataclass
class SumQuantileLoss:
    q: float

    def __call__(self, axis: Optional[int] = None) -> DirectEvaluator:
        return DirectEvaluator(
            name=f"sum_quantile_loss[{self.q}]",
            stat=partial(quantile_loss, q=self.q),
            aggregate=Sum(axis=axis),
        )


@dataclass
class Coverage:
    q: float

    def __call__(self, axis: Optional[int] = None) -> DirectEvaluator:
        return DirectEvaluator(
            name=f"coverage[{self.q}]",
            stat=partial(coverage, q=self.q),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MAPE:
    """Mean Absolute Percentage Error"""

    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> DirectEvaluator:
        return DirectEvaluator(
            name="MAPE",
            stat=partial(
                absolute_percentage_error, forecast_type=self.forecast_type
            ),
            aggregate=Mean(axis=axis),
        )


@dataclass
class SMAPE:
    """Symmetric Mean Absolute Percentage Error"""

    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> DirectEvaluator:
        return DirectEvaluator(
            name="sMAPE",
            stat=partial(
                symmetric_absolute_percentage_error,
                forecast_type=self.forecast_type,
            ),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MSIS:
    """Mean Scaled Interval Score"""

    alpha: float = 0.05

    def __call__(self, axis: Optional[int] = None) -> DirectEvaluator:
        return DirectEvaluator(
            name="MSIS",
            stat=partial(scaled_interval_score, alpha=self.alpha),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MASE:
    """Mean Absolute Scaled Error"""

    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> DirectEvaluator:
        return DirectEvaluator(
            name="MASE",
            stat=partial(
                absolute_scaled_error, forecast_type=self.forecast_type
            ),
            aggregate=Mean(axis=axis),
        )


@dataclass
class ND:
    """Normalized Deviation"""

    forecast_type: str = "0.5"

    @staticmethod
    def normalized_deviation(
        sum_absolute_error: np.ndarray, sum_absolute_label: np.ndarray
    ) -> np.ndarray:
        return sum_absolute_error / sum_absolute_label

    def __call__(self, axis: Optional[int] = None) -> DerivedEvaluator:
        return DerivedEvaluator(
            name="ND",
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

    def __call__(self, axis: Optional[int] = None) -> DerivedEvaluator:
        return DerivedEvaluator(
            name="RMSE",
            evaluators={
                "mean_squared_error": MSE(forecast_type=self.forecast_type)(
                    axis=axis
                )
            },
            post_process=self.root_mean_squared_error,
        )


@dataclass
class NRMSE:
    """RMSE, normalized by the mean absolute label"""

    forecast_type: str = "mean"

    @staticmethod
    def normalize_root_mean_squared_error(
        root_mean_squared_error: np.ndarray, mean_absolute_label: np.ndarray
    ) -> np.ndarray:
        return root_mean_squared_error / mean_absolute_label

    def __call__(self, axis: Optional[int] = None) -> DerivedEvaluator:
        return DerivedEvaluator(
            name="NRMSE",
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

    def __call__(self, axis: Optional[int] = None) -> DerivedEvaluator:
        return DerivedEvaluator(
            name=f"weighted_sum_quantile_loss[{self.q}]",
            evaluators={
                "sum_quantile_loss": SumQuantileLoss(q=self.q)(axis=axis),
                "sum_absolute_label": sum_absolute_label(axis=axis),
            },
            post_process=self.weight_sum_quantile_loss,
        )


@dataclass
class MeanSumQuantileLoss:
    quantile_levels: Collection[float]

    @staticmethod
    def mean(**quantile_losses: np.ndarray) -> np.ndarray:
        stacked_quantile_losses = np.stack(
            [quantile_loss for quantile_loss in quantile_losses.values()],
            axis=0,
        )
        return np.ma.mean(stacked_quantile_losses, axis=0)

    def __call__(self, axis: Optional[int] = None) -> DerivedEvaluator:
        return DerivedEvaluator(
            name="mean_sum_quantile_loss",
            evaluators={
                f"quantile_loss[{q}]": SumQuantileLoss(q=q)(axis=axis)
                for q in self.quantile_levels
            },
            post_process=self.mean,
        )


@dataclass
class MeanWeightedSumQuantileLoss:
    quantile_levels: Collection[float]

    @staticmethod
    def mean(**quantile_losses: np.ndarray) -> np.ndarray:
        stacked_quantile_losses = np.stack(
            [quantile_loss for quantile_loss in quantile_losses.values()],
            axis=0,
        )
        return np.ma.mean(stacked_quantile_losses, axis=0)

    def __call__(self, axis: Optional[int] = None) -> DerivedEvaluator:
        return DerivedEvaluator(
            name="mean_weighted_sum_quantile_loss",
            evaluators={
                f"quantile_loss[{q}]": WeightedSumQuantileLoss(q=q)(axis=axis)
                for q in self.quantile_levels
            },
            post_process=self.mean,
        )


@dataclass
class MAECoverage:
    quantile_levels: Collection[float]

    @staticmethod
    def mean(
        quantile_levels: Collection[float], **coverages: np.ndarray
    ) -> np.ndarray:
        intermediate_result = np.stack(
            [np.abs(coverages[f"coverage[{q}]"] - q) for q in quantile_levels],
            axis=0,
        )
        return np.ma.mean(intermediate_result, axis=0)

    def __call__(self, axis: Optional[int] = None) -> DerivedEvaluator:
        return DerivedEvaluator(
            name="MAE_coverage",
            evaluators={
                f"coverage[{q}]": Coverage(q=q)(axis=axis)
                for q in self.quantile_levels
            },
            post_process=partial(
                self.mean, quantile_levels=self.quantile_levels
            ),
        )


@dataclass
class OWA:
    """Overall Weighted Average"""

    forecast_type: str = "0.5"

    @staticmethod
    def calculate_OWA(
        smape: np.ndarray,
        smape_naive2: np.ndarray,
        mase: np.ndarray,
        mase_naive2: np.ndarray,
    ) -> np.ndarray:
        return 0.5 * (smape / smape_naive2 + mase / mase_naive2)

    def __call__(self, axis: Optional[int] = None) -> DerivedEvaluator:
        return DerivedEvaluator(
            name="OWA",
            evaluators={
                "smape": SMAPE(forecast_type=self.forecast_type)(axis=axis),
                "smape_naive2": SMAPE(forecast_type="naive_2")(axis=axis),
                "mase": MASE(forecast_type=self.forecast_type)(axis=axis),
                "mase_naive2": MASE(forecast_type="naive_2")(axis=axis),
            },
            post_process=self.calculate_OWA,
        )
