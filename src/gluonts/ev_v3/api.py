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

import copy
from abc import abstractmethod, ABC
from typing import Dict, Collection, Union, Iterator

import numpy as np

from gluonts.dataset import DataEntry
from gluonts.model.forecast import Quantile, Forecast


class Metric(ABC):
    def __init__(self):
        self._name = None

    @property
    def name(self) -> str:
        # for parameters given in __init__ (if any), return a *unique* name
        if self._name is None:
            self._name = self.get_name()
        return self._name

    def get(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        if self.name not in data:
            data[self.name] = self.calculate(data)

        return data[self.name]

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def calculate(self, data: dict) -> np.ndarray:
        pass


class PrimitiveForecastBatch:
    def __init__(self, forecasts: Collection[Forecast]):
        self.forecasts = forecasts
        self.batch_size = len(forecasts)

    def quantile(self, q: Union[Quantile, float, str]) -> np.ndarray:
        result = [forecast.quantile(q) for forecast in self.forecasts]
        return np.stack(result)

    @property
    def median(self):
        return self.quantile(Quantile.parse(0.5))

    @property
    def mean(self) -> np.ndarray:
        result = [forecast.mean for forecast in self.forecasts]
        return np.stack(result)

    def __len__(self):
        return self.batch_size



def evaluate_batch(
    metrics: Collection[Metric], input_data: Dict[str, np.ndarray]
):
    requested_metrics = set(metric.name for metric in metrics)

    result_data = copy.deepcopy(input_data)
    for metric in metrics:
        metric.get(result_data)

    result = {
        metric_name: result_data[metric_name]
        for metric_name in result_data
        if metric_name in requested_metrics
    }
    result["entry_count"] = len(input_data["forecast"])

    return result


def aggregate(total_result, batch_result, metrics):
    for metric in metrics:
        if hasattr(metric, "axis") and metric.axis == 0:
            # TODO: Aggregating partial (batched) results is the tricky part.
            #  For now, batch results are always aggregated using mean which
            #  doesn't make sense in many cases.
            #  For very large dataset and metrics like MSE, it seems like
            #  we have to keep around all values in memory anyway...

            c1 = total_result["entry_count"]
            c2 = batch_result["entry_count"]

            weighted_total = total_result[metric.name] * c1
            weighted_new = batch_result[metric.name] * c2

            total_result[metric.name] = (weighted_total + weighted_new) / (
                c1 + c2
            )
        else:
            total_result[metric.name] = np.concatenate(
                (total_result[metric.name], batch_result[metric.name])
            )

    total_result["entry_count"] += batch_result["entry_count"]

    return total_result


def finalize(eval_results: dict):
    return eval_results  # some final adjustments could be made here


def evaluate(metrics: Collection[Metric], input_batches: Iterator[dict]):
    eval_total = evaluate_batch(metrics, next(input_batches))
    for batch in input_batches:
        eval_partial = evaluate_batch(metrics, batch)
        eval_total = aggregate(eval_total, eval_partial, metrics)

    result = finalize(eval_total)
    return result


def get_input_batches(
    dataset_it: Iterator[DataEntry],
    forecast_it: Iterator[Forecast],
    batch_size: int,
) -> Iterator[dict]:
    done = False
    while not done:
        target_batch = []
        past_data_batch = []
        forecast_batch = []

        try:
            for _ in range(batch_size):
                data_entry = next(dataset_it)
                forecast = next(forecast_it)

                target_batch.append(
                    data_entry["target"][-forecast.prediction_length :]
                )
                past_data_batch.append(
                    data_entry["target"][: -forecast.prediction_length]
                )
                forecast_batch.append(forecast)
        except StopIteration:
            done = True

        if len(target_batch) > 0:
            input_batch = {
                "target": np.stack(target_batch),
                "past_data": np.stack(past_data_batch),
                "forecast": PrimitiveForecastBatch(forecast_batch),
            }
            yield input_batch
