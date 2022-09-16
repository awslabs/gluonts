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
import warnings
from abc import abstractmethod, ABC
from typing import Dict, Collection, Union, Iterator, Optional

import numpy as np


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

    # TODO: actually use these methods
    def aggr_batch(self, batch_1, batch_2):
        # batch_1 always gets updated with values from batch_2
        # TODO: allow user to have control over this
        pass

    def _aggr_batch_sum(self, batch_1, batch_2):
        batch_1[self.name] = batch_1[self.name] + batch_2[self.name]

    def _aggr_batch_mean(self, batch_1, batch_2):
        c1 = batch_1["entry_count"]
        c2 = batch_2["entry_count"]

        weighted_1 = batch_1[self.name] * c1
        weighted_2 = batch_2[self.name] * c2

        batch_1[self.name] = (weighted_1 + weighted_2) / (c1 + c2)


def evaluate(metrics: Collection[Metric], input_data: Dict[str, np.ndarray]):
    """
    This function evaluates `metrics` on `input_data`.
    `input_data` might be an entire dataset or a batch.
    """
    requested_metrics = set(metric.name for metric in metrics)

    result_data = copy.deepcopy(input_data)
    for metric in metrics:
        metric.get(result_data)

    result = {
        metric_name: result_data[metric_name]
        for metric_name in result_data
        if metric_name in requested_metrics
    }
    result["entry_count"] = len(input_data["forecast"])

    return result


def aggregate_batches(total_result, batch_result, metrics):
    for metric in metrics:
        if hasattr(metric, "axis") and metric.axis == 0:
            warnings.warn(
                "Batched calculation for metrics using axis=0"
                f" isn't supported, skipping metric '{metric.name}'"
            )
            total_result.pop(metric.name, None)
        else:
            total_result[metric.name] = np.concatenate(
                (total_result[metric.name], batch_result[metric.name])
            )

    total_result["entry_count"] += batch_result["entry_count"]

    return total_result


def finalize_evaluation(eval_results: dict):
    # some final adjustments / additions to the metric results could be made here
    return eval_results


def evaluate_batches(
    metrics: Collection[Metric], input_batches: Iterator[dict]
):
    eval_total = evaluate(metrics, next(input_batches))
    for batch in input_batches:
        eval_partial = evaluate(metrics, batch)
        eval_total = aggregate_batches(eval_total, eval_partial, metrics)

    result = finalize_evaluation(eval_total)
    return result
