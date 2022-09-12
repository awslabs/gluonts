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

import copy
from collections import UserDict
from typing import Callable, Dict, Collection
from dataclasses import dataclass

import numpy as np


@dataclass
class Input:
    name: str

    def resolve_base_dependencies(self, data):
        assert self.name in data, f"Missing data for input '{self.name}'"


@dataclass
class BaseMetric:
    name: str
    fn: Callable
    dependencies: tuple

    def apply(self, data):
        data[self.name] = self.fn(
            **{
                dependency.name: data[dependency.name]
                for dependency in self.dependencies
            },
        )

    def resolve_base_dependencies(self, data):
        if self.name not in data:
            for dependency in self.dependencies:
                dependency.resolve_base_dependencies(data)

            # base metrics are calculated right away, we don't need to wait for anything else
            self.apply(data)

    def apply_aggregate(self, data, axis):
        return


@dataclass
class AggregateMetric:
    name: str
    fn: Callable
    dependencies: tuple

    def resolve_base_dependencies(self, data):
        for dependency in self.dependencies:
            dependency.resolve_base_dependencies(data)

        data[self.name] = self  # insert a placeholder for later use

    def apply_aggregate(self, data, axis):
        for d in self.dependencies:
            d.apply_aggregate(data, axis)

        data[self.name] = self.fn(
            **{
                dependency.name: data[dependency.name]
                for dependency in self.dependencies
            },
            axis=axis,
        )


# these decorators only work for metrics without extra parameters
def metric(*dependencies, name=None):
    def decorator(fn):
        return BaseMetric(
            name=name or fn.__name__,
            dependencies=dependencies,
            fn=fn,
        )

    return decorator


def aggregate(*dependencies, name=None):
    def decorator(fn):
        return AggregateMetric(
            name=name or fn.__name__,
            dependencies=dependencies,
            fn=fn,
        )

    return decorator


@dataclass
class EvalResult(UserDict):
    data: dict
    select: set

    def get_base_metrics(self):
        return {
            metric_name: self[metric_name]
            for metric_name in self.select
            if isinstance(self[metric_name], np.ndarray)
        }

    def get_aggregate_metrics(self, axis=None):
        result = copy.deepcopy(self.data)

        for metric_name in self.select:
            if isinstance(self[metric_name], AggregateMetric):
                self[metric_name].apply_aggregate(result, axis=axis)

        return {
            metric_name: result[metric_name]
            for metric_name in self.select
            if isinstance(self[metric_name], AggregateMetric)
        }


def evaluate(metrics, data: Dict[str, np.ndarray]):
    result = copy.deepcopy(data)

    for metric in metrics:
        # note that aggregated metrics are calculated separately later on
        metric.resolve_base_dependencies(result)

    return EvalResult(result, select=set(metric.name for metric in metrics))
