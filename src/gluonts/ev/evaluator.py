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

from collections import defaultdict
from dataclasses import dataclass
from typing import List

from toolz import keyfilter

from .api import Stat, Aggregation


def resolve_dependencies(entities):
    def resolve(entity):
        entities = {}

        for dep in map(resolve, entity.dependencies):
            entities.update(dep)

        entities[entity.name] = entity

        return entities

    result = {}

    for entity in map(resolve, entities):
        result.update(entity)

    return result


class Evaluator:
    def __init__(self, metrics):
        # target metrics
        self.target_metrics = set(metric.name for metric in metrics)

        resolved = resolve_dependencies(metrics).values()

        self.stats = [
            metric for metric in resolved if isinstance(metric, Stat)
        ]

        self.aggregations = [
            metric for metric in resolved if isinstance(metric, Aggregation)
        ]

    def apply(self, ts, forecast):
        metrics = {}

        for stat in self.stats:
            stat.apply(ts, forecast, metrics)

        return Metrics(
            data=metrics,
            aggregation_fns=self.aggregations,
            target_metrics=self.target_metrics,
        )


@dataclass
class Metrics:
    data: dict
    aggregation_fns: List[Aggregation]
    target_metrics: set

    def aggregate(self, axis=None):
        aggregations = {}

        for aggr in self.aggregation_fns:
            aggr.apply_aggregate(self.data, aggregations, axis)

        return Metrics(aggregations, self.aggregation_fns, self.target_metrics)

    def group_by(self, key):
        keys = list(key(self.data))

        result = {k: defaultdict(list) for k in keys}

        for metrcic_name, values in self.data.items():
            for k, v in zip(keys, values):
                result[k][metrcic_name].append(v)

        return result

    def select(self):
        return keyfilter(lambda name: name in self.target_metrics, self.data)
