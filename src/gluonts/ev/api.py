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
from typing import Callable, Dict, Iterator, Collection, List, Optional
import numpy as np

from gluonts.model.forecast import Forecast
from gluonts.dataset.split import TestData
from gluonts.ev.helpers import EvalData, create_eval_data
from gluonts.ev.aggregations import Aggregation


def gather_inputs(
    test_data: TestData,
    forecasts: Iterator[Forecast],
    quantile_levels: Collection[float],
    batch_size: int = 64,
) -> Iterator[EvalData]:
    """Collect relevant data as NumPy arrays to evaluate the next batch.

    The number of entries in `test_data` and `forecasts` must be equal.
    """
    inputs = iter(test_data.input)
    labels = iter(test_data.label)

    input_data = []
    label_data = []
    forecast_data = {
        "mean": [],
        **{str(q): [] for q in quantile_levels},
    }

    entry_counter = 0
    for _ in range(len(test_data)):
        forecast_entry = next(forecasts)
        forecast_data["mean"].append(forecast_entry.mean)
        for q in quantile_levels:
            forecast_data[str(q)].append(forecast_entry.quantile(q))

        input_data.append(next(inputs))
        label_data.append(next(labels))

        entry_counter += 1
        if entry_counter % batch_size == 0:
            yield create_eval_data(input_data, label_data, forecast_data)

            input_data.clear()
            label_data.clear()
            for key in forecast_data:
                forecast_data[key].clear()

    if entry_counter % batch_size != 0:
        yield create_eval_data(input_data, label_data, forecast_data)


class DataProbe:
    """A DataProbe gathers all quantile forecasts required for an evaluation.

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


class Metric:
    def __call__(self, axis: Optional[int] = None) -> "MetricEvaluator":
        raise NotImplementedError


class MetricEvaluator:
    def update(self, data: EvalData) -> None:
        raise NotImplementedError

    def get(self) -> np.ndarray:
        raise NotImplementedError

    def reset() -> None:
        raise NotImplementedError


@dataclass
class StandardMetricEvaluator(MetricEvaluator):
    """A "standard metric" consists of a metric function and aggregation strategy."""

    map: Callable
    aggregate: Aggregation

    def update(self, data: EvalData) -> None:
        self.aggregate.step(self.map(data))

    def get(self) -> np.ndarray:
        return self.aggregate.get()

    def reset(self) -> None:
        self.aggregate.reset()


@dataclass
class DerivedMetricEvaluator(MetricEvaluator):
    """A "derived metric" depends on the prior calculation of "standard metrics"."""

    metrics: Dict[str, StandardMetricEvaluator]
    post_process: Callable

    def update(self, data: EvalData) -> None:
        for metric in self.metrics.values():
            metric.update(data)

    def get(self) -> np.ndarray:
        return self.post_process(
            **{name: metric.get() for name, metric in self.metrics.items()}
        )

    def reset(self) -> None:
        for metric_evaluator in self.metrics.values():
            metric_evaluator.reset()


class Evaluator:
    def __init__(self) -> None:
        # TODO: better naming!
        self.metric_evaluators: dict[str, MetricEvaluator] = dict()

    def add_metric(self, metrics: Metric, axis: Optional[int] = None) -> None:
        self.add_metrics([metrics], axis)

    def add_metrics(
        self, metrics: Collection[Metric], axis: Optional[int] = None
    ) -> None:
        for metric in metrics:
            metric_evaluator = metric(axis=axis)
            metric_name = f"{metric.__class__.__name__}[axis={axis}]"
            self.metric_evaluators[metric_name] = metric_evaluator

    def get_required_quantile_levels(self, test_data: TestData):
        data_probe = DataProbe(test_data)
        for metric_evaluator in self.metric_evaluators.values():
            metric_evaluator.reset()
            metric_evaluator.update(data_probe)
            metric_evaluator.get()
        return data_probe.required_quantile_forecasts

    def evaluate(
        self, test_data: TestData, forecasts: Iterator[Forecast]
    ) -> EvalData:
        quantile_levels = self.get_required_quantile_levels(test_data)

        batches = gather_inputs(
            test_data=test_data,
            forecasts=forecasts,
            quantile_levels=quantile_levels,
        )

        # only NumPy arrays are used from here on

        for metric_evaluator in self.metric_evaluators.values():
            metric_evaluator.reset()

        for data in batches:
            for metric_evaluator in self.metric_evaluators.values():
                metric_evaluator.update(data)

        result = dict()
        for metric_name, metric_evaluator in self.metric_evaluators.items():
            result[metric_name] = metric_evaluator.get()
        return result
