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

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import (
    Collection,
    Optional,
    Callable,
    Mapping,
    Dict,
    List,
    Iterator,
)
from typing_extensions import Protocol, runtime_checkable, Self

import numpy as np

from .aggregations import Aggregation, Mean, Sum
from .stats import (
    error,
    absolute_error,
    absolute_label,
    absolute_percentage_error,
    absolute_scaled_error,
    coverage,
    quantile_loss,
    scaled_interval_score,
    scaled_quantile_loss,
    squared_error,
    symmetric_absolute_percentage_error,
    num_masked_target_values,
)


@dataclass
class MetricCollection:
    metrics: List[Metric]

    def update(self, data: Mapping[str, np.ndarray]) -> Self:
        """
        Update metrics using a single data instance.
        """

        for metric in self.metrics:
            metric.update(data)

        return self

    def update_all(self, stream: Iterator[Mapping[str, np.ndarray]]) -> Self:
        """
        Update metrics using a stream of data instances.
        """

        for element in stream:
            self.update(element)

        return self

    def get(self) -> Dict[str, np.ndarray]:
        return {metric.name: metric.get() for metric in self.metrics}


@dataclass
class Metric:
    name: str

    def update(self, data: Mapping[str, np.ndarray]) -> Self:
        """
        Update metric using a single data instance.
        """

        raise NotImplementedError

    def update_all(self, stream: Iterator[Mapping[str, np.ndarray]]) -> Self:
        """
        Update metric using a stream of data instances.
        """

        for element in stream:
            self.update(element)

        return self

    def get(self) -> np.ndarray:
        raise NotImplementedError


@dataclass
class DirectMetric(Metric):
    """
    A Metric which uses a single function and aggregation strategy.
    """

    stat: Callable
    aggregate: Aggregation

    def update(self, data: Mapping[str, np.ndarray]) -> Self:
        self.aggregate.step(self.stat(data))

        return self

    def get(self) -> np.ndarray:
        return self.aggregate.get()


@dataclass
class DerivedMetric(Metric):
    """
    A Metric that is computed using other metrics.

    A derived metric updates multiple, simpler metrics independently and in the
    end combines their results as defined in `post_process`.
    """

    metrics: Dict[str, Metric]
    post_process: Callable

    def update(self, data: Mapping[str, np.ndarray]) -> Self:
        for evaluator in self.metrics.values():
            evaluator.update(data)

        return self

    def get(self) -> np.ndarray:
        return self.post_process(
            **{
                name: evaluator.get()
                for name, evaluator in self.metrics.items()
            }
        )


@runtime_checkable
class MetricDefinition(Protocol):
    def __call__(self, axis: Optional[int] = None) -> Metric:
        raise NotImplementedError


class BaseMetricDefinition:
    def __call__(self, axis):
        raise NotImplementedError()

    def __add__(self, other) -> MetricDefinitionCollection:
        if isinstance(other, MetricDefinitionCollection):
            return other + self

        return MetricDefinitionCollection([self, other])

    def add(self, *others):
        for other in others:
            self = self + other

        return self


@dataclass
class MetricDefinitionCollection(BaseMetricDefinition):
    metrics: List[BaseMetricDefinition]

    def __call__(self, axis: Optional[int] = None) -> MetricCollection:
        return MetricCollection([metric(axis=axis) for metric in self.metrics])

    def __add__(self, other) -> MetricDefinitionCollection:
        if isinstance(other, MetricDefinitionCollection):
            return MetricDefinitionCollection([*self.metrics, *other.metrics])

        return MetricDefinitionCollection([*self.metrics, other])


class MeanAbsoluteLabel(BaseMetricDefinition):
    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name="mean_absolute_label",
            stat=absolute_label,
            aggregate=Mean(axis=axis),
        )


mean_absolute_label = MeanAbsoluteLabel()


class SumAbsoluteLabel(BaseMetricDefinition):
    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name="sum_absolute_label",
            stat=absolute_label,
            aggregate=Sum(axis=axis),
        )


sum_absolute_label = SumAbsoluteLabel()


class SumNumMaskedTargetValues(BaseMetricDefinition):
    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name="sum_num_masked_target_values",
            stat=num_masked_target_values,
            aggregate=Sum(axis=axis),
        )


sum_num_masked_target_values = SumNumMaskedTargetValues()


@dataclass
class SumError(BaseMetricDefinition):
    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name=f"sum_error[{self.forecast_type}]",
            stat=partial(error, forecast_type=self.forecast_type),
            aggregate=Sum(axis=axis),
        )


sum_error = SumError()


@dataclass
class SumAbsoluteError(BaseMetricDefinition):
    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name=f"sum_absolute_error[{self.forecast_type}]",
            stat=partial(absolute_error, forecast_type=self.forecast_type),
            aggregate=Sum(axis=axis),
        )


sum_absolute_error = SumAbsoluteError()


@dataclass
class MAE(BaseMetricDefinition):
    """
    Mean Absolute Error.
    """

    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name=f"MAE[{self.forecast_type}]",
            stat=partial(absolute_error, forecast_type=self.forecast_type),
            aggregate=Mean(axis=axis),
        )


mae = MAE()


@dataclass
class MSE(BaseMetricDefinition):
    """
    Mean Squared Error.
    """

    forecast_type: str = "mean"

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name=f"MSE[{self.forecast_type}]",
            stat=partial(squared_error, forecast_type=self.forecast_type),
            aggregate=Mean(axis=axis),
        )


mse = MSE()


@dataclass
class SumQuantileLoss(BaseMetricDefinition):
    q: float

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name=f"sum_quantile_loss[{self.q}]",
            stat=partial(quantile_loss, q=self.q),
            aggregate=Sum(axis=axis),
        )


@dataclass
class Coverage(BaseMetricDefinition):
    q: float

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name=f"coverage[{self.q}]",
            stat=partial(coverage, q=self.q),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MAPE(BaseMetricDefinition):
    """
    Mean Absolute Percentage Error.
    """

    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name=f"MAPE[{self.forecast_type}]",
            stat=partial(
                absolute_percentage_error, forecast_type=self.forecast_type
            ),
            aggregate=Mean(axis=axis),
        )


mape = MAPE()


@dataclass
class SMAPE(BaseMetricDefinition):
    """
    Symmetric Mean Absolute Percentage Error.
    """

    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name=f"sMAPE[{self.forecast_type}]",
            stat=partial(
                symmetric_absolute_percentage_error,
                forecast_type=self.forecast_type,
            ),
            aggregate=Mean(axis=axis),
        )


smape = SMAPE()


@dataclass
class MSIS(BaseMetricDefinition):
    """
    Mean Scaled Interval Score.
    """

    alpha: float = 0.05

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name="MSIS",
            stat=partial(scaled_interval_score, alpha=self.alpha),
            aggregate=Mean(axis=axis),
        )


msis = MSIS()


@dataclass
class MASE(BaseMetricDefinition):
    """
    Mean Absolute Scaled Error.
    """

    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name=f"MASE[{self.forecast_type}]",
            stat=partial(
                absolute_scaled_error, forecast_type=self.forecast_type
            ),
            aggregate=Mean(axis=axis),
        )


mase = MASE()


@dataclass
class MeanScaledQuantileLoss(BaseMetricDefinition):
    q: float

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name=f"mean_scaled_quantile_loss[{self.q}]",
            stat=partial(scaled_quantile_loss, q=self.q),
            aggregate=Mean(axis=axis),
        )


@dataclass
class ND(BaseMetricDefinition):
    """
    Normalized Deviation.
    """

    forecast_type: str = "0.5"

    @staticmethod
    def normalized_deviation(
        sum_absolute_error: np.ndarray, sum_absolute_label: np.ndarray
    ) -> np.ndarray:
        return sum_absolute_error / sum_absolute_label

    def __call__(self, axis: Optional[int] = None) -> DerivedMetric:
        return DerivedMetric(
            name=f"ND[{self.forecast_type}]",
            metrics={
                "sum_absolute_error": SumAbsoluteError(
                    forecast_type=self.forecast_type
                )(axis=axis),
                "sum_absolute_label": sum_absolute_label(axis=axis),
            },
            post_process=self.normalized_deviation,
        )


nd = ND()


@dataclass
class RMSE(BaseMetricDefinition):
    """
    Root Mean Squared Error.
    """

    forecast_type: str = "mean"

    @staticmethod
    def root_mean_squared_error(mean_squared_error: np.ndarray) -> np.ndarray:
        return np.sqrt(mean_squared_error)

    def __call__(self, axis: Optional[int] = None) -> DerivedMetric:
        return DerivedMetric(
            name=f"RMSE[{self.forecast_type}]",
            metrics={
                "mean_squared_error": MSE(forecast_type=self.forecast_type)(
                    axis=axis
                )
            },
            post_process=self.root_mean_squared_error,
        )


rmse = RMSE()


@dataclass
class NRMSE(BaseMetricDefinition):
    """
    RMSE, normalized by the mean absolute label.
    """

    forecast_type: str = "mean"

    @staticmethod
    def normalize_root_mean_squared_error(
        root_mean_squared_error: np.ndarray, mean_absolute_label: np.ndarray
    ) -> np.ndarray:
        return root_mean_squared_error / mean_absolute_label

    def __call__(self, axis: Optional[int] = None) -> DerivedMetric:
        return DerivedMetric(
            name=f"NRMSE[{self.forecast_type}]",
            metrics={
                "root_mean_squared_error": RMSE(
                    forecast_type=self.forecast_type
                )(axis=axis),
                "mean_absolute_label": mean_absolute_label(axis=axis),
            },
            post_process=self.normalize_root_mean_squared_error,
        )


nrmse = NRMSE()


@dataclass
class WeightedSumQuantileLoss(BaseMetricDefinition):
    q: float

    @staticmethod
    def weight_sum_quantile_loss(
        sum_quantile_loss: np.ndarray, sum_absolute_label: np.ndarray
    ) -> np.ndarray:
        return sum_quantile_loss / sum_absolute_label

    def __call__(self, axis: Optional[int] = None) -> DerivedMetric:
        return DerivedMetric(
            name=f"weighted_sum_quantile_loss[{self.q}]",
            metrics={
                "sum_quantile_loss": SumQuantileLoss(q=self.q)(axis=axis),
                "sum_absolute_label": sum_absolute_label(axis=axis),
            },
            post_process=self.weight_sum_quantile_loss,
        )


@dataclass
class MeanSumQuantileLoss(BaseMetricDefinition):
    quantile_levels: Collection[float]

    @staticmethod
    def mean(**quantile_losses: np.ndarray) -> np.ndarray:
        stacked_quantile_losses = np.stack(
            [quantile_loss for quantile_loss in quantile_losses.values()],
            axis=0,
        )
        return np.mean(stacked_quantile_losses, axis=0)

    def __call__(self, axis: Optional[int] = None) -> DerivedMetric:
        return DerivedMetric(
            name="mean_sum_quantile_loss",
            metrics={
                f"quantile_loss[{q}]": SumQuantileLoss(q=q)(axis=axis)
                for q in self.quantile_levels
            },
            post_process=self.mean,
        )


@dataclass
class MeanWeightedSumQuantileLoss(BaseMetricDefinition):
    quantile_levels: Collection[float]

    @staticmethod
    def mean(**quantile_losses: np.ndarray) -> np.ndarray:
        stacked_quantile_losses = np.stack(
            [quantile_loss for quantile_loss in quantile_losses.values()],
            axis=0,
        )
        return np.mean(stacked_quantile_losses, axis=0)

    def __call__(self, axis: Optional[int] = None) -> DerivedMetric:
        return DerivedMetric(
            name="mean_weighted_sum_quantile_loss",
            metrics={
                f"quantile_loss[{q}]": WeightedSumQuantileLoss(q=q)(axis=axis)
                for q in self.quantile_levels
            },
            post_process=self.mean,
        )


@dataclass
class AverageMeanScaledQuantileLoss(BaseMetricDefinition):
    quantile_levels: Collection[float]

    @staticmethod
    def mean(**quantile_losses: np.ndarray) -> np.ndarray:
        stacked_quantile_losses = np.stack(
            [quantile_loss for quantile_loss in quantile_losses.values()],
            axis=0,
        )
        return np.mean(stacked_quantile_losses, axis=0)

    def __call__(self, axis: Optional[int] = None) -> DerivedMetric:
        return DerivedMetric(
            name="average_mean_scaled_quantile_loss",
            metrics={
                f"mean_scaled_quantile_loss[{q}]": MeanScaledQuantileLoss(q=q)(
                    axis=axis
                )
                for q in self.quantile_levels
            },
            post_process=self.mean,
        )


@dataclass
class MAECoverage(BaseMetricDefinition):
    quantile_levels: Collection[float]

    @staticmethod
    def mean(
        quantile_levels: Collection[float], **coverages: np.ndarray
    ) -> np.ndarray:
        intermediate_result = np.stack(
            [np.abs(coverages[f"coverage[{q}]"] - q) for q in quantile_levels],
            axis=0,
        )
        return np.mean(intermediate_result, axis=0)

    def __call__(self, axis: Optional[int] = None) -> DerivedMetric:
        return DerivedMetric(
            name="MAE_coverage",
            metrics={
                f"coverage[{q}]": Coverage(q=q)(axis=axis)
                for q in self.quantile_levels
            },
            post_process=partial(
                self.mean, quantile_levels=self.quantile_levels
            ),
        )


@dataclass
class OWA(BaseMetricDefinition):
    """
    Overall Weighted Average.
    """

    forecast_type: str = "0.5"

    @staticmethod
    def calculate_OWA(
        smape: np.ndarray,
        smape_naive2: np.ndarray,
        mase: np.ndarray,
        mase_naive2: np.ndarray,
    ) -> np.ndarray:
        return 0.5 * (smape / smape_naive2 + mase / mase_naive2)

    def __call__(self, axis: Optional[int] = None) -> DerivedMetric:
        return DerivedMetric(
            name=f"OWA[{self.forecast_type}]",
            metrics={
                "smape": SMAPE(forecast_type=self.forecast_type)(axis=axis),
                "smape_naive2": SMAPE(forecast_type="naive_2")(axis=axis),
                "mase": MASE(forecast_type=self.forecast_type)(axis=axis),
                "mase_naive2": MASE(forecast_type="naive_2")(axis=axis),
            },
            post_process=self.calculate_OWA,
        )


owa = OWA()
