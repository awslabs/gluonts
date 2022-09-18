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

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Dict, Optional, Union, List
import numpy as np

from gluonts.model import Forecast
from gluonts.model.forecast import Quantile


class Metric(ABC):
    def __init__(self):
        self._name = None

    @property
    def name(self) -> str:
        # for parameters given in __init__ (if any), return a **unique** name
        if self._name is None:
            self._name = self.get_name()
        return self._name

    @abstractmethod
    def get_name(self) -> str:
        pass


class BaseMetric(Metric, ABC):
    def get(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        if self.name not in data:
            data[self.name] = self.calculate(data)

        return data[self.name]

    @abstractmethod
    def calculate(self, data: dict) -> np.ndarray:
        pass


class AggregateMetric(Metric, ABC):
    def get(
        self, data: Dict[str, np.ndarray], axis: Optional[int] = None
    ) -> np.ndarray:
        if self.name not in data:
            data[self.name] = self.calculate(data, axis)

        return data[self.name]

    @abstractmethod
    def calculate(self, data: dict, axis: Optional[int]) -> np.ndarray:
        pass


@dataclass
class EvalResult:
    base_metrics: Dict[str, np.ndarray]
    metrics_per_entry: Dict[str, np.ndarray]
    metric_per_timestamp: Dict[str, np.ndarray]
    global_metrics: Dict[str, np.ndarray]
    custom_metrics: Dict[str, np.ndarray]


class BatchedForecasts:
    def __init__(self, forecasts: List[Forecast]):
        self.forecasts = forecasts

    @property
    def mean(self) -> np.ndarray:
        return np.stack([forecast.mean for forecast in self.forecasts])

    def quantile(self, q: Union[Quantile, float, str]) -> np.ndarray:
        return np.stack([forecast.quantile(q) for forecast in self.forecasts])

    def __len__(self):
        return len(self.forecasts)

    def __getitem__(self, idx):
        return self.forecasts[idx]


def get_standard_type(value: Union[Quantile, float, str]):
    if value == "mean":
        return "mean"

    if value == "median":
        value = "0.5"
    return f"{Quantile.parse(value).name}"
