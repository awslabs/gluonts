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

from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Optional,
    Protocol,
    runtime_checkable,
)

import numpy as np

from gluonts.dataset.split import TestData
from ..ev_data_preparation.data_construction import construct_data
from gluonts.model.predictor import Predictor
from .aggregations import Aggregation


@runtime_checkable
class Metric(Protocol):
    def __call__(self, axis: Optional[int] = None) -> "Evaluator":
        raise NotImplementedError


class Evaluator:
    def evaluate(
        self, test_data: TestData, predictor: Predictor, **predictor_kwargs
    ) -> np.ndarray:
        data_batches = construct_data(test_data, predictor, **predictor_kwargs)

        for data in data_batches:
            self.update(data)

        return self.get()

    def update(self, data: Dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def get(self) -> np.ndarray:
        raise NotImplementedError


@dataclass
class DirectEvaluator(Evaluator):
    """An Evaluator which uses a single function and aggregation strategy."""

    map: Callable
    aggregate: Aggregation

    def update(self, data: Dict[str, np.ndarray]) -> None:
        self.aggregate.step(self.map(data))

    def get(self) -> np.ndarray:
        return self.aggregate.get()


@dataclass
class DerivedEvaluator(Evaluator):
    """An Evaluator for metrics that are derived by other metrics.

    A derived metric updates multiple, simpler metrics independently and in
    the end combines their results as defined in `post_process`."""

    evaluators: Dict[str, Evaluator]
    post_process: Callable

    def update(self, data: Dict[str, np.ndarray]) -> None:
        for metric in self.evaluators.values():
            metric.update(data)

    def get(self) -> np.ndarray:
        return self.post_process(
            **{name: metric.get() for name, metric in self.evaluators.items()}
        )


@dataclass
class MetricGroup:
    """Allows feeding in data once and calculating multiple metrics"""

    metric_evaluators: Dict[str, Evaluator] = field(default_factory=dict)

    def add_metric(
        self, metric_name: str, metric: Metric, axis: Optional[int] = None
    ) -> None:
        self.add_metrics({metric_name: metric}, axis)

    def add_metrics(
        self, metrics: Dict[str, Metric], axis: Optional[int] = None
    ) -> None:
        for metric_name, metric in metrics.items():
            assert (
                metric_name not in self.metric_evaluators
            ), f"Metric name '{metric_name}' is not unique"

            metric_evaluator = metric(axis=axis)
            self.metric_evaluators[metric_name] = metric_evaluator

    def evaluate(
        self, test_data: TestData, predictor: Predictor, **predictor_kwargs
    ) -> Dict[str, np.ndarray]:
        data_batches = construct_data(test_data, predictor, **predictor_kwargs)

        for data in data_batches:
            for metric_evaluator in self.metric_evaluators.values():
                metric_evaluator.update(data)

        result = {
            metric_name: metric_evaluator.get()
            for metric_name, metric_evaluator in self.metric_evaluators.items()
        }
        return result
