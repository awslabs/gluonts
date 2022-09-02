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


import numpy as np
from dataclasses import dataclass
from typing import Dict, Collection, Iterator, List

from toolz import keyfilter

from .api import Metric, PointMetric
from .metrics import MSE, Mape, AbsError, Error
from ..dataset.split import TestDataset
from ..model import Forecast


def resolve_dependencies(metrics: Collection[Metric]) -> Collection[Metric]:
    # note: only considers metric.name
    # we don't worry about aggregation_name here because the only
    # "dependency" for an aggregation is the underlying metric
    def resolve(metric):
        metrics = {}

        for dep in map(resolve, metric.dependencies):
            metrics.update(dep)

        metrics[metric.name] = metric

        return metrics

    result = {}

    for metric in map(resolve, metrics):
        result.update(metric)

    return result.values()


def topo_sort_metrics(metrics: Collection[Metric]) -> List[Metric]:
    return list(metrics)  # todo: actually sort


@dataclass
class Metrics:
    data: Dict[str, Dict[str, np.ndarray]]
    target_metrics: Dict[str, List[str]]
    metadata: dict

    def get_point_metrics(self) -> Dict[str, np.ndarray]:
        return keyfilter(
            lambda name: name in self.target_metrics["point"], self.data
        )

    def get_local_metrics(self) -> Dict[str, np.ndarray]:
        return keyfilter(
            lambda name: name in self.target_metrics["local"], self.data
        )

    def get_global_metrics(self) -> Dict[str, float]:
        return keyfilter(
            lambda name: name in self.target_metrics["global"], self.data
        )

    def get_all(self):
        all_target_metrics = dict()
        for key, value in self.data.items():
            if (
                key in self.target_metrics["point"]
                or key in self.target_metrics["local"]
                or key in self.target_metrics["global"]
            ):
                all_target_metrics[key] = value
        return all_target_metrics


class Evaluator:
    _default_metrics = (
        MSE(),
        Error(),
        # AbsError(aggr="sum"),  # todo: make point metric aggregatable
        MSE(aggr="mean"),
        Mape(aggr="mean"),
    )

    def __init__(self, metrics: Collection[Metric] = _default_metrics) -> None:
        self.target_metrics = {"point": [], "local": [], "global": []}
        self.aggregations = []
        for metric in metrics:
            if isinstance(metric, PointMetric):
                self.target_metrics["point"].append(metric.name)
            elif metric.can_aggregate:
                self.target_metrics["global"].append(metric.aggregation_name)
                self.aggregations.append(metric)
            else:
                self.target_metrics["local"].append(metric.name)

        required_metrics = resolve_dependencies(metrics)
        self.required_metrics = topo_sort_metrics(
            required_metrics
        )  # aggregations will be calculated later

    def apply(
        self, test_pairs: TestDataset, forecasts: Iterator[Forecast]
    ) -> Metrics:
        metrics_data = {metric.name: [] for metric in self.required_metrics}
        metadata = {"item_id": [], "start": []}

        test_pairs_iter = iter(test_pairs)
        for index in range(len(test_pairs.dataset)):
            input_data, label = next(test_pairs_iter)
            metadata["item_id"].append(input_data["item_id"])
            metadata["start"].append(input_data["start"])
            forecast = next(forecasts)

            latest_metrics_data = dict()
            for metric in self.required_metrics:
                value = metric.get(
                    input_data, label, forecast, latest_metrics_data
                )
                metrics_data[metric.name].append(value)
                latest_metrics_data[metric.name] = value

        metrics_data_np = {
            key: np.stack(value, axis=0) for key, value in metrics_data.items()
        }

        aggregations = dict()
        for metric in self.aggregations:
            aggregations[metric.aggregation_name] = metric.get_aggregate(
                metrics_data[metric.name]
            )

        return Metrics(
            data={**metrics_data_np, **aggregations},
            target_metrics=self.target_metrics,
            metadata=metadata,
        )
