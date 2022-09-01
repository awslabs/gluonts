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
from abc import ABC
from typing import Collection, Union, Tuple, Optional, Callable, Dict

import numpy as np

from gluonts.dataset import DataEntry
from gluonts.model import Forecast


class Metric(ABC):
    name: str
    dependencies: Collection["Metric"] = ()

    def __init__(self):
        self.aggregation_name = None
        self.can_aggregate = False

    def get(
        self,
        input_data: DataEntry,
        label: DataEntry,
        forecast: Forecast,
        metrics: Dict[str, float],
    ) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def get_aggregate(self, metrics: Dict[str, np.ndarray]) -> float:
        raise NotImplementedError


# todo: the only reason for having this class currently is that it's clear what `get` returns - is there a better way?
class PointMetric(Metric, ABC):
    def get(
        self,
        input_data: DataEntry,
        label: DataEntry,
        forecast: Forecast,
        metrics: Dict[str, float],
    ) -> np.ndarray:
        raise NotImplementedError


class LocalMetric(Metric, ABC):
    def _get_aggregate_mean(self, metrics: Dict[str, np.ndarray]) -> float:
        return np.mean(metrics[self.name]).item()

    def _get_aggregate_sum(self, metrics: Dict[str, np.ndarray]) -> float:
        return np.sum(metrics[self.name]).item()

    def __init__(
        self,
        aggr: Optional[
            Union[str, Callable[[Dict[str, np.ndarray]], float]]
        ] = None,
    ):
        super().__init__()
        if aggr is not None:
            self.can_aggregate = True
            if aggr == "mean":
                self.get_aggregate = self._get_aggregate_mean
                aggregation_name = "mean"
            elif aggr == "sum":
                self.get_aggregate = self._get_aggregate_sum
                aggregation_name = "sum"
            else:
                self.get_aggregate = aggr
                aggregation_name = aggr.__name__
            self.aggregation_name = f"{self.name}_{aggregation_name}"

    # todo: again, this is just here to explicitly state the return type
    def get(
        self,
        input_data: DataEntry,
        label: DataEntry,
        forecast: Forecast,
        metrics: Dict[str, float],
    ) -> float:
        raise NotImplementedError
