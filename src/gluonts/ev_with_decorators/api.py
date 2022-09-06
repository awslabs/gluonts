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

    def apply(self, data):
        assert self.name in data, f"Missing data for input '{self.name}'"


@dataclass
class Input:
    name: str

    def apply(self, data):
        assert self.name in data, f"Missing data for input '{self.name}'"


@dataclass
class Metric:
    name: str
    dependencies: tuple
    fn: Callable

    def apply(self, data):
        if self.name not in data:
            for dependency in self.dependencies:
                dependency.apply(data)

            data[self.name] = self.fn(
                **{
                    dependency.name: data[dependency.name]
                    for dependency in self.dependencies
                }
            )


def metric(*dependencies, name=None):
    def decorator(fn):
        return Metric(
            name=name or fn.__name__,
            dependencies=dependencies,
            fn=fn,
        )

    return decorator


@dataclass
class EvalResult(UserDict):
    data: dict
    select: Collection

    def get_all(self):
        return {
            metric_name: self.data[metric_name] for metric_name in self.select
        }


def evaluate(metrics, data: Dict[str, np.ndarray]):
    result = copy.deepcopy(data)

    for metric in metrics:
        metric.apply(result)

    return EvalResult(result, select=list(metric.name for metric in metrics))
