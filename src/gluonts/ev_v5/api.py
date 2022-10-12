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
from typing import Callable, Iterator, Optional
import numpy as np

# BATCH STRATEGIES (Concat, Sum, Mean)


@dataclass
class Concat:
    results: list = field(default_factory=list)

    def step(self, values):
        self.results.append(values)

    def aggregate(self):
        return np.concatenate(self.results)


@dataclass
class Sum:
    axis: Optional[int]
    results: list = field(default_factory=list)

    def step(self, values):
        self.results.append(np.sum(values, axis=self.axis))

    def aggregate(self):
        if self.axis is None or self.axis == 0:
            return np.sum(self.results, axis=self.axis)

        return np.concatenate(self.results)


@dataclass
class Mean:
    axis: Optional[int]
    results: list = field(default_factory=list)
    n: int = 0

    def step(self, values):
        self.results.append(np.sum(values, axis=self.axis))

        if self.axis is None:
            self.n += values.size
        else:
            self.n += values.shape[self.axis]

    def aggregate(self):
        if self.axis is None or self.axis == 0:
            return np.sum(self.results, axis=self.axis) / self.n

        return np.concatenate(self.results) / self.n


# "simple metrics" includes all metrics which are not derived metrics
class SimpleMetric:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

        # the following will be set by concrete subclass
        self.super_aggregate = None
        self.metric_fn = None

    def step(self, data):
        func_res = self.metric_fn(data, **self.kwargs)
        self.super_aggregate.step(func_res)

    def get(self):
        return self.super_aggregate.aggregate()


def evaluate(
    batches: Iterator,
    metric: SimpleMetric,
):
    for batch in batches:
        metric.step(batch)

    return metric.get()
