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

from gluonts.dataset import DataEntry
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
        if key == "batch_size":
            return 1
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
    input_it: Iterator[DataEntry],
    label_it: Iterator[DataEntry],
    forecast_it: Iterator[Forecast],
    quantile_levels: Collection[float],
    batch_size: int,
):
    input_data = []
    label_data = []

    forecast_data = {
        "mean": [],
        **{str(q): [] for q in quantile_levels},
    }

    actual_batch_size = 0  # is less than batch_size if there's no more data
    for _ in range(batch_size):
        try:
            forecast_entry = next(forecast_it)
            forecast_data["mean"].append(forecast_entry.mean)
            for q in quantile_levels:
                forecast_data[str(q)].append(forecast_entry.quantile(q))

            input_data.append(next(input_it))
            label_data.append(next(label_it))

            actual_batch_size += 1
        except StopIteration:
            break

    if actual_batch_size == 0:
        return dict()

    input_data = {
        "batch_size": actual_batch_size,
        "input": np.stack([entry["target"] for entry in input_data]),
        "label": np.stack([entry["target"] for entry in label_data]),
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


def evaluate_batch(
    data: Union[dict, DataProbe],
    metric_specs: Collection[MetricSpec],
) -> dict:
    # only NumPy arrays needed for actual evaluation
    batch_result = {
        metric.name: metric.fn(data=data, **metric.parameters)
        for metric in metric_specs
    }
    batch_result["batch_size"] = data["batch_size"]

    return batch_result


def aggregate_batch_results(
    batch_result_1, batch_result_2, metric_specs: Collection[MetricSpec]
):
    # assumption: keys in batches are the same

    aggregate_result = dict()

    for metric_spec in metric_specs:
        name = metric_spec.name
        parameters = metric_spec.parameters

        count_1 = batch_result_1["batch_size"]
        count_2 = batch_result_2["batch_size"]

        if (
            "axis" in parameters
            and parameters["axis"] == 0
            or np.ndim(batch_result_1[name]) == 0
        ):
            # taking the mean might not always make sense
            # TODO: include in MetricSpec objects HOW to aggregate batches
            aggregate_result[name] = (
                count_1 * batch_result_1[name] + count_2 * batch_result_2[name]
            ) / (count_1 + count_2)
        else:
            aggregate_result[name] = np.concatenate(
                (batch_result_1[name], batch_result_2[name])
            )

    aggregate_result["batch_size"] = (
        batch_result_1["batch_size"] + batch_result_2["batch_size"]
    )
    return aggregate_result


def evaluate(
    test_data: TestData,
    forecast_it: Iterator[Forecast],
    metric_specs: Collection[MetricSpec] = default_metric_specs,
    batch_size: int = 64,
):
    # determine required quantile levels
    data_probe = DataProbe(test_data)
    evaluate_batch(data_probe, metric_specs)
    quantile_levels = data_probe.required_quantile_forecasts

    input_it = iter(test_data.input)
    label_it = iter(test_data.label)

    eval_batches = []
    while True:
        batch_data = gather_inputs(
            input_it=input_it,
            label_it=label_it,
            forecast_it=forecast_it,
            quantile_levels=quantile_levels,
            batch_size=batch_size,
        )

        if len(batch_data) == 0:
            break

        batch_result = evaluate_batch(batch_data, metric_specs)
        eval_batches.append(batch_result)

    eval_result = eval_batches[0]
    for i in range(1, len(eval_batches)):
        eval_result = aggregate_batch_results(
            eval_result, eval_batches[i], metric_specs
        )

    return eval_result
