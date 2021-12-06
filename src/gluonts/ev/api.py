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


from typing import Collection


class Stat:
    name: str
    dependencies: Collection["Stat"] = ()

    def apply(self, ts, forecast, metrics):
        metrics[self.name] = self.get(ts, forecast, metrics)

    def get(self, ts, forecast, metrics):
        raise NotImplementedError


class Aggregation:
    name: str
    dependencies: Collection["Aggregation"]

    def apply_aggregate(self, metrics, aggregations, axis):
        aggregations[self.name] = self.aggregate(metrics, aggregations, axis)

    def aggregate(self, metrics, aggregations, axis):
        raise NotImplementedError


class Metric(Stat, Aggregation):
    def get_aggr(self, metric):
        raise NotImplementedError

    def aggregate(self, metrics, aggregations, axis):
        return self.get_aggr(metrics[self.name], axis)


class DerivedMetric(Aggregation):
    def aggregate(self, metrics, aggregations, axis):
        return self.derive_from(aggregations)
