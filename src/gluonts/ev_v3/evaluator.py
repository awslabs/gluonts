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
from typing import Iterator, Collection, Optional, Callable, Union

import numpy as np

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import TestDataset
from gluonts.ev_v3.api import (
    Metric,
    AggregateMetric,
    EvalResult,
    BaseMetric,
    BatchedForecasts,
)
from gluonts.ev_v3.metrics import (
    MSE,
    AbsPredictionTarget,
    SeasonalError,
    ND,
    QuantileLoss,
    MAPE,
    SMAPE,
    MASE,
    MSIS,
    Coverage,
    RMSE,
    Error,
)
from gluonts.model import Forecast


class NewEvaluator:
    _default_quantiles = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

    _default_base_metrics = (
        Error("mean"),
        AbsPredictionTarget(),
        *(QuantileLoss(quantile=q) for q in _default_quantiles),
        *(Coverage(quantile=q) for q in _default_quantiles),
    )

    _default_timestamp_metrics = (MSE(),)

    def __init__(self, batch_size=2048):
        self.batch_size = batch_size

    def _get_input_batches(
        self, dataset: TestDataset, forecasts: Iterator[Forecast]
    ) -> Iterator[dict]:
        done = False
        dataset_it = iter(dataset)

        while not done:
            prediction_target_batch = []
            input_data_batch = []
            forecast_batch = []

            try:
                for _ in range(self.batch_size):
                    input_data, prediction_target = next(dataset_it)
                    input_data_batch.append(input_data[FieldName.TARGET])
                    prediction_target_batch.append(
                        prediction_target[FieldName.TARGET]
                    )
                    forecast_batch.append(next(forecasts))
            except StopIteration:
                done = True

            if len(forecast_batch) > 0:
                input_batch = {
                    "InputData": np.stack(input_data_batch),
                    "PredictionTarget": np.stack(prediction_target_batch),
                    "Forecast": BatchedForecasts(forecast_batch),
                }
                yield input_batch

    def _aggregate_batches(self, batches):
        if len(batches) == 0:
            return dict()

        # assumption: dicts in each batch have the same keys
        metric_names = set(batches[0].keys())

        return {
            metric_name: np.concatenate(
                [batch[metric_name] for batch in batches]
            )
            for metric_name in metric_names
        }

    def __call__(
        self,
        dataset: TestDataset,
        forecasts: Iterator[Forecast],
        freq: str,
        base_metrics: Collection[BaseMetric] = _default_base_metrics,
        metrics_per_entry: Optional[Collection[AggregateMetric]] = None,
        metrics_per_timestamp: Collection[
            AggregateMetric
        ] = _default_timestamp_metrics,
        custom_metrics: Optional[Collection[Union[Callable, Metric]]] = None,
    ):
        if metrics_per_entry is None:
            metrics_per_entry = (
                MSE(),
                RMSE(),
                MAPE(),
                SMAPE(),
                ND(),
                SeasonalError(freq=freq),
                MASE(freq=freq),
                MSIS(freq=freq),
            )

        base_metrics_batches = []
        metrics_per_entry_batches = []
        metrics_per_timestamp_batches = []
        global_metrics_batches = []  # TODO
        custom_metrics_batches = []

        for data in self._get_input_batches(dataset, forecasts):
            # note: the data dictionary is written to within each method
            # but only requested data is returned

            base_metrics_batches.append(
                {metric.name: metric.get(data) for metric in base_metrics}
            )

            data_before_aggregates = copy.deepcopy(data)
            metrics_per_entry_batches.append(
                {
                    metric.name: metric.get(data, 1)
                    for metric in metrics_per_entry
                }
            )

            data = data_before_aggregates  # axis is not part of metric names -> reset
            metrics_per_timestamp_batches.append(
                {
                    metric.name: metric.get(data, 1)
                    for metric in metrics_per_timestamp
                }
            )

            custom_metrics_batch = dict()
            for metric_fn in custom_metrics:
                input_data = data["InputData"]
                prediction_target = data["PredictionTarget"]
                forecast = data["Forecast"]

                batch_size = len(
                    input_data
                )  # could be less than self.batch_size

                custom_metric_result = []
                for idx in range(batch_size):
                    custom_metric_result.append(
                        metric_fn(
                            input_data[idx],
                            prediction_target[idx],
                            forecast[idx],
                        )
                    )
                custom_metrics_batch[metric_fn.__name__] = np.stack(
                    custom_metric_result
                )

            custom_metrics_batches.append(custom_metrics_batch)

        return EvalResult(
            *(
                self._aggregate_batches(metrics_batches)
                for metrics_batches in (
                    base_metrics_batches,
                    metrics_per_entry_batches,
                    metrics_per_timestamp_batches,
                    global_metrics_batches,
                    custom_metrics_batches,
                )
            )
        )
