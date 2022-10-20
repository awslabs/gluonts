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

from collections import ChainMap
from typing import Collection, Dict, Iterator, Optional
import numpy as np
from .stats import seasonal_error

from gluonts.model.forecast import Forecast
from gluonts.dataset.split import TestData
from .api import Metric, MetricEvaluator
from .metrics import (
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
    MeanSquaredError,
    MeanScaledIntervalScore,
    NormalizedDeviation,
    NormalizedRootMeanSquaredError,
    RootMeanSquaredError,
    SymmetricMeanAbsolutePercentageError,
    AbsoluteErrorSum,
    AbsoluteLabelMean,
    AbsoluteLabelSum,
    Coverage,
    QuantileLoss,
    WeightedQuantileLoss,
)


class DataBatch:
    """Used to add batch dimension"""

    def __init__(self, values) -> None:
        self.values = values

    def __getitem__(self, name):
        return np.array([self.values[name]])


class Evaluator:
    def __init__(self) -> None:
        # TODO: better naming!
        self.metric_evaluators: Dict[str, MetricEvaluator] = dict()

    def add_metric(self, metrics: Metric, axis: Optional[int] = None) -> None:
        self.add_metrics([metrics], axis)

    def add_metrics(
        self, metrics: Collection[Metric], axis: Optional[int] = None
    ) -> None:
        for metric in metrics:
            metric_evaluator = metric(axis=axis)
            metric_name = f"{metric.__class__.__name__}[axis={axis}]"
            self.metric_evaluators[metric_name] = metric_evaluator

    def add_default_metrics(self):
        metrics_per_entry = [
            MeanSquaredError(),
            AbsoluteErrorSum(),
            AbsoluteLabelSum(),
            AbsoluteLabelMean(),
            MeanAbsoluteScaledError(),
            MeanAbsolutePercentageError(),
            SymmetricMeanAbsolutePercentageError(),
            NormalizedDeviation(),
            MeanScaledIntervalScore(),
            *[QuantileLoss(q=q) for q in (0.1, 0.5, 0.9)],
            *[Coverage(q=q) for q in (0.1, 0.5, 0.9)],
        ]
        global_metrics = [
            MeanSquaredError(),
            AbsoluteErrorSum(),
            AbsoluteLabelSum(),
            AbsoluteLabelMean(),
            MeanAbsoluteScaledError(),
            MeanAbsolutePercentageError(),
            SymmetricMeanAbsolutePercentageError(),
            NormalizedDeviation(),
            MeanScaledIntervalScore(),
            *[QuantileLoss(q=q) for q in (0.1, 0.5, 0.9)],
            *[WeightedQuantileLoss(q=q) for q in (0.1, 0.5, 0.9)],
            *[Coverage(q=q) for q in (0.1, 0.5, 0.9)],
            RootMeanSquaredError(),
            NormalizedRootMeanSquaredError(),
        ]

        self.add_metrics(metrics_per_entry, axis=1)
        self.add_metrics(global_metrics, axis=None)

    def get_batches(self, test_data: TestData, forecasts: Iterator[Forecast]):
        seasonality = 1  # TODO: use actual seasonality

        for test_entry, forecast in zip(test_data, forecasts):
            batching_used = not isinstance(
                forecast, Forecast
            )  # as opposed to ForecastBatch

            input, label = test_entry

            non_forecast_data = {
                "label": label["target"],
                "seasonal_error": seasonal_error(
                    input["target"], seasonality=seasonality
                ),
            }

            batch_data = ChainMap(non_forecast_data, forecast)

            if batching_used:
                yield batch_data
            else:
                yield DataBatch(batch_data)

    def evaluate(
        self, test_data: TestData, forecasts: Iterator[Forecast]
    ) -> Dict[str, np.ndarray]:
        batches = self.get_batches(
            test_data=test_data,
            forecasts=forecasts,
        )

        for metric_evaluator in self.metric_evaluators.values():
            metric_evaluator.reset()

        for data in batches:
            for metric_evaluator in self.metric_evaluators.values():
                metric_evaluator.update(data)

        result = dict()
        for metric_name, metric_evaluator in self.metric_evaluators.items():
            result[metric_name] = metric_evaluator.get()
        return result
