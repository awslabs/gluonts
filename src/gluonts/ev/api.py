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
from typing import Callable, Dict, Optional

import numpy as np

from .aggregations import Aggregation


class Metric:
    def __call__(self, axis: Optional[int] = None) -> "MetricEvaluator":
        raise NotImplementedError


class MetricEvaluator:
    def update(self, data: Dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def get(self) -> np.ndarray:
        raise NotImplementedError

    def reset() -> None:
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
