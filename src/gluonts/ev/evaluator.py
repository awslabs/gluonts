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
    Collection,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

import numpy as np

from gluonts.dataset.split import TestData
from gluonts.ev_data_preparation.data_construction import construct_data
from gluonts.model.predictor import Predictor
from .aggregations import Aggregation


@runtime_checkable
class Metric(Protocol):
    def __call__(self, axis: Optional[int] = None) -> "Evaluator":
        raise NotImplementedError


@dataclass
class Evaluator:
    name: str

    def evaluate(
        self,
        test_data: TestData,
        predictor: Predictor,
        ignore_invalid_values: bool = True,
        **predictor_kwargs,
    ) -> np.ndarray:
        data_batches = construct_data(
            test_data,
            predictor,
            ignore_invalid_values=ignore_invalid_values,
            **predictor_kwargs,
        )

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
    """An Evaluator for metrics that are derived from other metrics.

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
    """Allows calculating multiple metrics while feeding in data only once"""

    axis: Optional[int] = None
    evaluators: Dict[str, Evaluator] = field(default_factory=dict)

    def add_metric(self, metric: Metric, name: Optional[str] = None) -> None:
        if name is not None:
            self.add_named_metrics({name: metric})
        else:
            self.add_metrics([metric])

    def add_metrics(self, metrics: Collection[Metric]) -> None:
        for metric in metrics:
            evaluator = metric(axis=self.axis)
            self._validate_name(evaluator.name)
            self.evaluators[evaluator.name] = evaluator

    def add_named_metrics(self, metrics: Dict[str, Metric]) -> None:
        for name, metric in metrics.items():
            self._validate_name(name)
            evaluator = metric(axis=self.axis, name=name)
            self.evaluators[name] = evaluator

    def evaluate(
        self,
        test_data: TestData,
        predictor: Predictor,
        ignore_invalid_values: bool = True,
        **predictor_kwargs,
    ) -> np.ndarray:
        data_batches = construct_data(
            test_data,
            predictor,
            ignore_invalid_values=ignore_invalid_values,
            **predictor_kwargs,
        )
        for data in data_batches:
            for evaluator in self.evaluators.values():
                evaluator.update(data)

        result = {
            metric_name: evaluator.get()
            for metric_name, evaluator in self.evaluators.items()
        }
        return result

    def _validate_name(self, name: str) -> None:
        assert (
            name not in self.evaluators
        ), f"Evaluator name '{name}' is not unique"
