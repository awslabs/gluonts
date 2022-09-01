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
from typing import Collection, Optional, Union, Callable

from dataclasses import dataclass

import numpy as np

from .api import PointMetric, LocalMetric


# --- POINT METRICS ---


class AbsTarget(PointMetric):
    name: str = "abs_target"

    def get(self, input_data, label, forecast, metrics) -> np.ndarray:
        return np.abs(label["target"])


class Error(PointMetric):
    name: str = "error"

    def get(self, input_data, label, forecast, metrics) -> np.ndarray:
        return label["target"] - forecast.median


class AbsError(PointMetric):
    name: str = "abs_error"
    dependencies = [Error()]

    def get(self, input_data, label, forecast, metrics) -> np.ndarray:
        return np.abs(metrics["error"])


class SquaredError(PointMetric):
    name: str = "squared_error"
    dependencies = [Error()]

    def get(self, input_data, label, forecast, metrics) -> np.ndarray:
        return np.square(metrics["error"])


# --- LOCAL METRICS ---


class MSE(LocalMetric):
    name: str = "mse"
    dependencies = [SquaredError()]

    def get(self, input_data, label, forecast, metrics) -> float:
        return np.mean(metrics["squared_error"]).item()


class AbsTargetSum(LocalMetric):
    name: str = "abs_target_sum"
    dependencies = [AbsTarget()]

    def get(self, input_data, label, forecast, metrics) -> float:
        return np.sum(metrics["abs_target"]).item()


class AbsErrorSum(LocalMetric):
    name: str = "abs_error_sum"
    dependencies = [AbsError()]

    def get(self, input_data, label, forecast, metrics) -> float:
        return np.sum(metrics["abs_error"][forecast.item_id]).item()


class Mape(LocalMetric):
    name: str = "mape"
    dependencies = [AbsError(), AbsTarget()]

    def get(self, input_data, label, forecast, metrics) -> float:
        return np.mean(metrics["abs_error"] / metrics["abs_target"]).item()
