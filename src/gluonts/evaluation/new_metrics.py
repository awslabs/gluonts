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
from functools import partial
from itertools import chain
from typing import List

import numpy as np


class Stat:
    name: str
    dependencies = ()

    def apply(self, x):
        raise NotImplementedError


class Metric(Stat):
    name: str
    dependencies = ()

    def apply(self, x):
        x[self.name] = self.loss(x)

    def get_aggr(self, metrics, aggrs, axis):
        return self.aggregate(metrics[self.name], axis=axis)

    def loss(self, x):
        raise NotImplementedError

    def aggregate(self, metrics, axis=None):
        raise NotImplementedError


class DerivedMetric:
    name: str
    dependencies = ()

    def get_aggr(self, metrics, aggrs, axis):
        return self.get(aggrs)

    def get(self, aggrs):
        raise NotImplementedError


class Loss(Stat):
    name: str = "loss"

    def apply(self, x):
        x["loss"] = x["target"] - x["forecast"]


class AggrMean:
    def aggregate(self, loss, axis=None):
        return np.mean(loss, axis=axis)


class AggrSum:
    def aggregate(self, loss, axis=None):
        return np.sum(loss, axis=axis)


class AbsTargetSum(AggrSum, Metric):
    name: str = "abs_target_sum"

    def loss(self, x):
        return x["target"]


@dataclass
class QuantileLoss(AggrSum, Metric):
    q: float

    @property
    def name(self):
        return f"quantile[{self.q}]"

    def loss(self, x):
        return 2 * np.abs(
            x["loss"]
            * ((x["target"] <= x["forecast"].quantile(self.q)) - self.q)
        )


@dataclass
class WeightedQuantileLoss(DerivedMetric):
    quantile: float

    @property
    def name(self):
        return f"wQuantileLoss[{self.quantile}]"

    @property
    def dependencies(self):
        return [QuantileLoss(self.quantile), AbsTargetSum()]

    def get(self, metrics):
        ql = QuantileLoss(self.quantile)

        return metrics[ql.name] / metrics["abs_target_sum"]


@dataclass
class MeanWeightedQuantileLoss(DerivedMetric):
    quantiles: List[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    name: str = "mean_wQuantileLoss"

    @property
    def dependencies(self):
        return list(map(WeightedQuantileLoss, self.quantiles))

    def get(self, metrics):
        return np.mean(
            [metrics[quantile.name] for quantile in self.dependencies],
            axis=0,
        )


def merge(a, b):
    """Only merge items from b if they are not in a.

    This is needed so that the order of `a` is not altered.
    """
    for key, val in dict.items(b):
        if key not in a:
            a[key] = val

    return a


def resolve(entity):
    entities = {}

    for dep in map(resolve, entity.dependencies):
        merge(entities, dep)

    entities[entity.name] = entity

    return entities


def get_order(entities):
    result = {}

    for entity in entities:
        resolved = resolve(entity)
        merge(result, resolved)

    return result


def calc_aggrs(aggr_fns, data, axis=None):
    result = {}

    for aggr in aggr_fns:
        result[aggr.name] = aggr.get_aggr(data, result, axis=axis)

    return result


def calc_losses(loss_fns, data):
    result = {}

    for loss in loss_fns:
        loss.apply(data)

    return data


def select_metrics(names, data):
    return {name: value for name, value in dict.items(data) if name in names}


def get_evaluator(metrics):
    ops = get_order(metrics)

    losses = [op for op in ops.values() if isinstance(op, (Metric, Stat))]

    aggrs = [
        op for op in ops.values() if isinstance(op, (Metric, DerivedMetric))
    ]

    return (
        partial(calc_losses, losses),
        partial(calc_aggrs, aggrs),
        partial(select_metrics, [metric.name for metric in metrics]),
    )


import pandas as pd

data = {
    "target": np.array(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
        ]
    ),
    "forecast": np.array(
        [
            [2, 2, 2, 5, 5],
            [1, 2, 3, 4, 5],
        ]
    ),
}

from typing import List


class Evaluator:
    metrics: List[Metric]

    def __init__(self, metrics):
        self.metrics = metrics
        self._operations = get_order(metrics)
        stats = [
            op for op in self._operations.values() if isinstance(op, Stat)
        ]

        aggrs = [
            op
            for op in ops.values()
            if isinstance(op, (Metric, DerivedMetric))
        ]

    # def _get_losses(self, data):

    # def apply(self, data):


ev = Evaluator([MeanWeightedQuantileLoss()])


# get_losses, aggregate, select = get_evaluator([MeanWeightedQuantileLoss()])

# value_metrics = get_losses(data)
# item_metrics = aggregate(value_metrics, axis=1)
# aggr_metrics = aggregate(item_metrics)

# print(select(aggr_metrics))
