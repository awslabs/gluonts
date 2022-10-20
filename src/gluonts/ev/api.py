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

from collections import ChainMap
from dataclasses import dataclass, field
from typing import Callable, Collection, Dict, Iterator, Optional, Union

import numpy as np

from .stats import seasonal_error

from ..model.forecast import Forecast

from ..dataset.split import TestData

from .aggregations import Aggregation


class Metric:
    def __call__(self, axis: Optional[int] = None) -> "MetricEvaluator":
        raise NotImplementedError


class MetricEvaluator:
    def evaluate(
        self, batches: Iterator[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        self.reset()

        for data in batches:
            self.update(data)

        return self.get()

    def update(self, data: Dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def get(self) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


@dataclass
class StandardMetricEvaluator(MetricEvaluator):
    """A "standard metric" consists of a metric function and aggregation strategy."""

    map: Callable
    aggregate: Aggregation

    def update(self, data: Dict[str, np.ndarray]) -> None:
        self.aggregate.step(self.map(data))

    def get(self) -> np.ndarray:
        return self.aggregate.get()

    def reset(self) -> None:
        self.aggregate.reset()


@dataclass
class DerivedMetricEvaluator(MetricEvaluator):
    """A "derived metric" depends on the prior calculation of "standard metrics"."""

    metrics: Dict[str, StandardMetricEvaluator]
    post_process: Callable

    def update(self, data: Dict[str, np.ndarray]) -> None:
        for metric in self.metrics.values():
            metric.update(data)

    def get(self) -> np.ndarray:
        return self.post_process(
            **{name: metric.get() for name, metric in self.metrics.items()}
        )

    def reset(self) -> None:
        for metric_evaluator in self.metrics.values():
            metric_evaluator.reset()


@dataclass
class MultiMetricEvaluator(MetricEvaluator):
    """Allows feeding in data once and calculating multiple metrics"""

    metric_evaluators: Dict[str, MetricEvaluator] = field(default_factory=dict)

    def add_metric(self, metrics: Metric, axis: Optional[int] = None) -> None:
        self.add_metrics([metrics], axis)

    def add_metrics(
        self, metrics: Collection[Metric], axis: Optional[int] = None
    ) -> None:
        for metric in metrics:
            metric_evaluator = metric(axis=axis)
            metric_name = f"{metric.__class__.__name__}[axis={axis}]"
            self.metric_evaluators[metric_name] = metric_evaluator

    def update(self, data: Dict[str, np.ndarray]) -> None:
        for metric_evaluator in self.metric_evaluators.values():
            metric_evaluator.update(data)

    def get(self) -> np.ndarray:
        result = dict()
        for metric_name, metric_evaluator in self.metric_evaluators.items():
            result[metric_name] = metric_evaluator.get()
        return result

    def reset(self) -> None:
        for metric_evaluator in self.metric_evaluators.values():
            metric_evaluator.reset()


class EvaluationDataBatch:
    """Used to add batch dimension
    Should be replaced by a ForecastBatch eventually"""

    def __init__(self, values) -> None:
        self.values = values

    def __getitem__(self, name):
        return np.array([self.values[name]])


class ForecastBatch:
    pass  # see PR #2286


class TestDataBatch:
    pass  # TODO


def get_evaluation_data_batches(
    test_data: Union[TestData, TestDataBatch],
    forecasts: Iterator[Union[Forecast, ForecastBatch]],
):
    seasonality = 1  # TODO: use actual seasonality

    for input, label, forecast in zip(
        test_data.input, test_data.label, forecasts
    ):
        batching_used = isinstance(forecast, ForecastBatch)

        non_forecast_data = {
            "label": label["target"],
            "seasonal_error": seasonal_error(
                input["target"], seasonality=seasonality
            ),
        }
        joint_data = ChainMap(non_forecast_data, forecast)

        yield joint_data if batching_used else EvaluationDataBatch(joint_data)
