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
from .aggregations import Aggregation
from .stats import seasonal_error


def gather_inputs(
    test_data: TestData,
    forecasts: Iterator[Forecast],
    quantile_levels: Collection[float],
    batch_size: int = 64,
) -> Iterator[Dict[str, np.ndarray]]:
    """Collect relevant data as NumPy arrays to evaluate the next batch.

    The number of entries in `test_data` and `forecasts` must be equal.
    """

    def create_eval_data(
        seasonal_errors: np.ndarray, labels: List[np.ndarray], forecasts: Dict[str, np.ndarray]
    ):
        return {
            "seasonal_error": np.stack(seasonal_errors),
            "label": np.stack([entry["target"] for entry in labels]),
            **{name: np.stack(value) for name, value in forecasts.items()},
        }

    seasonality = 1  # TODO
    inputs = iter(test_data.input)
    labels = iter(test_data.label)

    seasonal_errors = []
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

        seasonal_error_value = seasonal_error(next(inputs)["target"], seasonality)
        seasonal_errors.append(seasonal_error_value)
        label_data.append(next(labels))

        entry_counter += 1
        if entry_counter % batch_size == 0:
            yield create_eval_data(seasonal_errors, label_data, forecast_data)

            seasonal_errors.clear()
            label_data.clear()
            for key in forecast_data:
                forecast_data[key].clear()

    if entry_counter % batch_size != 0:
        yield create_eval_data(seasonal_errors, label_data, forecast_data)


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
        self.input_shape = (1,)
        self.prediction_target_shape = (1,) + np.shape(label_sample["target"])

        self.required_quantile_forecasts = set()

    def __getitem__(self, key: str):
        if key == "batch_size":
            return 1
        if key == "seasonal_error":
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
    def update(self, data: Dict[str, np.ndarray]) -> None:
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

    def update(self, data: Dict[str, np.ndarray]) -> None:
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

    def update(self, data: Dict[str, np.ndarray]) -> None:
        for metric in self.metrics.values():
            metric.update(data)

    def get(self) -> np.ndarray:
        return self.post_process(
            **{name: metric.get() for name, metric in self.metrics.items()}
        )

    def reset(self) -> None:
        for metric_evaluator in self.metrics.values():
            metric_evaluator.reset()
