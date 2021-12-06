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

import numpy as np

from .api import Stat, Metric, DerivedMetric


class Loss(Stat):
    name: str = "loss"

    def get(self, ts, forecast, metrics):
        return ts["target"] - forecast.median


class AbsError(Metric):
    name: str = "abs_error"
    dependencies = [Loss()]

    def get(self, ts, forecast, metrics):
        return np.abs(metrics["loss"])

    def get_aggr(self, loss, axis):
        return np.sum(loss, axis=axis)


class AbsTarget(Stat):
    name: str = "abs_target"

    def get(self, ts, forecast, metrics):
        return np.abs(ts["target"])


class AbsTargetSum(Metric):
    name: str = "abs_target_sum"
    dependencies = [AbsTarget()]

    def get(self, ts, forecast, metrics):
        return metrics["abs_target"]

    def get_aggr(self, metric, axis):
        return np.sum(metric, axis=axis)


class Mape(Metric):
    name: str = "mape"
    dependencies = [AbsError(), AbsTarget()]

    def get(self, ts, forecast, metrics):
        return metrics["abs_error"] / metrics["abs_target"]

    def get_aggr(self, error, axis):
        return np.mean(error, axis=axis)


class ND(DerivedMetric):
    name: str = "ND"
    dependencies = [AbsError(), AbsTargetSum()]

    def derive_from(self, aggregations):
        return aggregations["abs_error"] / aggregations["abs_target_sum"]


@dataclass
class QuantileLoss(Metric):
    q: float

    dependencies = []

    @property
    def name(self):
        return f"quantile[{self.q}]"

    def get(self, ts, forecast, metrics):
        return 2 * np.abs(
            (ts.quantile(self.q) - forecast)
            * ((ts.quantile(self.q) <= forecast) - self.q)
        )
