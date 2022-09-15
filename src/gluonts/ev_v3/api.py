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
from abc import abstractmethod, ABC
from typing import Dict, Union, Collection

import numpy as np

from gluonts.model.forecast import Quantile


class Metric(ABC):
    def __init__(self):
        self._name = None

    @property
    def name(self) -> str:
        # for parameters given in __init__ (if any), return a *unique* name
        if self._name is None:
            self._name = self.get_name()
        return self._name

    def get(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        if self.name not in data:
            data[self.name] = self.calculate(data)

        return data[self.name]

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def calculate(self, data: dict) -> np.ndarray:
        pass


# mock ForecastBatch as described in PR #2286 with batch size = data_entry_count
# TODO: return actual values
class ForecastBatch:
    def __init__(self, prediction_length: int, batch_size: int):
        self.prediction_length = prediction_length
        self.batch_size = batch_size

    def quantile(self, q: Quantile) -> np.ndarray:
        return np.random.rand(self.batch_size, self.prediction_length)

    @property
    def median(self):
        return self.quantile(Quantile.parse(0.5))

    @property
    def mean(self) -> np.ndarray:
        return np.random.rand(self.batch_size, self.prediction_length)


def evaluate(metrics: Collection[Metric], data: Dict[str, np.ndarray]):
    requested_metrics = set(metric.name for metric in metrics)

    result = copy.deepcopy(data)
    for metric in metrics:
        metric.get(result)

    return {
        metric_name: result[metric_name]
        for metric_name in result
        if metric_name in requested_metrics
    }
