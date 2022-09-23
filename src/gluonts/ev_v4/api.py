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

from typing import Iterator, Callable, Collection, Union, Optional

import numpy as np

from gluonts.dataset.split import TestData
from gluonts.ev_23_09.metrics import mse
from gluonts.model import Forecast


class MetricSpec:
    def __init__(
        self, name: str, fn: Callable, parameters: Optional[dict] = None
    ):
        super().__init__()
        self.name = name
        self.fn = fn
        self.parameters = parameters or dict()


class DataProbe:
    # mission: gather all required quantile forecasts for metric calculation
    def __init__(self, test_data):
        input_sample, label_sample = next(iter(test_data))

        # use batch_size 1
        self.input_shape = (1,) + np.shape(input_sample["target"])
        self.label_shape = (1,) + np.shape(label_sample["target"])

        self.required_quantile_forecasts = set()

    def __getitem__(self, key: str):
        if key == "input":
            return np.random.rand(*self.input_shape)
        if key in ["label", "mean"]:
            return np.random.rand(*self.label_shape)

        try:
            float(key)
        except ValueError:
            raise ValueError(f"Unexpected input: {key}")

        self.required_quantile_forecasts.add(key)
        return np.random.rand(*self.label_shape)


def gather_inputs(
    test_data: TestData,
    forecast_it: Iterator[Forecast],
    metric_specs: Collection[MetricSpec],
):
    probe_data = DataProbe(test_data)
    evaluate(probe_data, metric_specs)
    quantile_levels = probe_data.required_quantile_forecasts

    forecast_data = {
        "mean": [],
        **{str(q): [] for q in quantile_levels},
    }
    for forecast in forecast_it:
        forecast_data["mean"].append(forecast.mean)
        for q in quantile_levels:
            forecast_data[str(q)] = forecast.quantile(q)

    input_data = {
        "input": np.stack([entry["target"] for entry in test_data.input]),
        "label": np.stack([entry["target"] for entry in test_data.label]),
        **{name: np.stack(value) for name, value in forecast_data.items()},
    }

    return input_data


default_metric_specs = (
    MetricSpec(
        name="MSE",
        fn=mse,
        parameters={"axis": 1},
    ),
    # TDOO: mimic old Evaluator class
)


def evaluate(
    data: Union[dict, DataProbe],
    metric_specs: Collection[MetricSpec] = default_metric_specs,
) -> dict:
    return {
        metric.name: metric.fn(data=data, **metric.parameters)
        for metric in metric_specs
    }
