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
from gluonts.ev_v4.metrics import mse
from gluonts.exceptions import GluonTSUserError
from gluonts.model import Forecast


class MetricSpec:
    def __init__(
        self,
        fn: Callable,
        name: Optional[str] = None,
        parameters: Optional[dict] = None,
    ):
        """`MetricSpec` objects specify how to use GluonTS metric functions.

        :param fn: metric function, for example from metrics.py
        :param name: custom metric name; defaults to fn.__name__
        :param parameters: all required parameters for fn except `data`
        """
        super().__init__()
        self.fn = fn
        self.parameters = parameters or dict()
        self.name = name or fn.__name__


class DataProbe:
    """ A DataProbe gathers all quantile forecasts required for an evaluation.

    This has the benefit that metric definitions can work independently of
    `Forecast` objects as all values in 'data' will be NumPy arrays.

    :raises ValueError: if a metric requests a key that can't be converted to
        float and isn't equal to "batch_size", "input", "label" or "mean"
    """
    def __init__(self, test_data: TestData):
        input_sample, label_sample = next(iter(test_data))
        # use batch_size 1
        self.input_shape = (1,) + np.shape(input_sample["target"])
        self.prediction_target_shape = (1,) + np.shape(label_sample["target"])

        self.required_quantile_forecasts = set()

    def __getitem__(self, key: str):
        if key == "batch_size":
            return 1
        if key == "input":
            return np.random.rand(*self.input_shape)
        if key in ["label", "mean"]:
            return np.random.rand(*self.prediction_target_shape)

        try:
            self.required_quantile_forecasts.add(float(key))
            return np.random.rand(*self.prediction_target_shape)
        except ValueError:
            raise ValueError(f"Unexpected input: {key}")


def gather_inputs(
    input_it: Iterator[DataEntry],
    label_it: Iterator[DataEntry],
    forecast_it: Iterator[Forecast],
    quantile_levels: Collection[float],
    batch_size: int,
):
    """Collect relevant data as NumPy arrays to evaluate the next batch."""
    input_data, label_data = [], []
    forecast_data = {
        "mean": [],
        **{str(q): [] for q in quantile_levels},
    }

    actual_batch_size = 0  # less than batch_size if iterators are used up
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
    # TODO: mimic old Evaluator class
)


def evaluate_batch(
    data: Union[dict, DataProbe],
    metric_specs: Collection[MetricSpec],
) -> dict:
    """Use `MetricSpec` objects to evaluate a single batch."""
    batch_result = {
        metric.name: metric.fn(data=data, **metric.parameters)
        for metric in metric_specs
    }
    batch_result["batch_size"] = data["batch_size"]

    return batch_result


def aggregate_batches(batch_1, batch_2, metric_specs: Collection[MetricSpec]):
    """ assumption: keys in batches are the same"""
    result = {"batch_size": batch_1["batch_size"] + batch_2["batch_size"]}

    for metric_spec in metric_specs:
        name = metric_spec.name
        params = metric_spec.parameters

        bs_1, bs_2 = batch_1["batch_size"], batch_2["batch_size"]
        values_1, values_2 = batch_1[name], batch_2[name]

        if np.ndim(values_1) == 0 or "axis" in params and params["axis"] == 0:
            # aggregate using mean
            weighted_sum = bs_1 * values_1 + bs_2 * values_2
            result[name] = weighted_sum / (bs_1 + bs_2)
        else:
            result[name] = np.concatenate((values_1, values_2))

    return result


def evaluate(
    test_data: TestData,
    forecast_it: Iterator[Forecast],
    metric_specs: Collection[MetricSpec] = default_metric_specs,
    batch_size: int = 64,
):
    """Evaluate metrics.

    :param test_data: time series to evaluate on
    :param forecast_it: Forecast objects corresponding to `test_data` entries
    :param metric_specs: GluonTS metrics to be evaluated
    :param batch_size: should be set small enough for data to fit into memory
    :raises GluonTSUserError: if metric names are used multiple times
    """
    # check for duplicate metric names
    names = set()
    for metric_spec in metric_specs:
        if metric_spec.name in names:
            raise GluonTSUserError(
                f"Metric name {metric_spec.name} is used multiple times"
            )
        names.add(metric_spec.name)

    # determine required quantile levels
    data_probe = DataProbe(test_data)
    evaluate_batch(data_probe, metric_specs)
    quantile_levels = data_probe.required_quantile_forecasts

    input_it = iter(test_data.input)
    label_it = iter(test_data.label)

    batches = []
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
        batches.append(batch_result)

    # reduce batch results to total result
    eval_result = batches[0]
    for i in range(1, len(batches)):
        eval_result = aggregate_batches(eval_result, batches[i], metric_specs)

    return eval_result
