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

from typing import Collection, Dict, Iterator, Optional

import numpy as np

from gluonts.model.forecast import Forecast
from gluonts.dataset.split import TestData
from .api import DataProbe, Metric, MetricEvaluator, gather_inputs
from .metrics import (
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    AbsErrorSum,
    AbsLabelMean,
    AbsLabelSum,
    Coverage,
    QuantileLoss,
    WeightedQuantileLoss,
)


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
            MSE(),
            AbsErrorSum(),
            AbsLabelSum(),
            AbsLabelMean(),
            MASE(),
            MAPE(),
            SMAPE(),
            ND(),
            MSIS(),
            *[QuantileLoss(q=q) for q in (0.1, 0.5, 0.9)],
            *[Coverage(q=q) for q in (0.1, 0.5, 0.9)],
        ]
        global_metrics = [
            MSE(),
            AbsErrorSum(),
            AbsLabelSum(),
            AbsLabelMean(),
            MASE(),
            MAPE(),
            SMAPE(),
            ND(),
            MSIS(),
            *[QuantileLoss(q=q) for q in (0.1, 0.5, 0.9)],
            *[WeightedQuantileLoss(q=q) for q in (0.1, 0.5, 0.9)],
            *[Coverage(q=q) for q in (0.1, 0.5, 0.9)],
            RMSE(),
            NRMSE(),
        ]

        self.add_metrics(metrics_per_entry, axis=1)
        self.add_metrics(global_metrics, axis=None)

    def get_required_quantile_levels(self, test_data: TestData):
        data_probe = DataProbe(test_data)
        for metric_evaluator in self.metric_evaluators.values():
            metric_evaluator.reset()
            metric_evaluator.update(data_probe)
            metric_evaluator.get()
        return data_probe.required_quantile_forecasts

    def evaluate(
        self, test_data: TestData, forecasts: Iterator[Forecast]
    ) -> Dict[str, np.ndarray]:
        quantile_levels = self.get_required_quantile_levels(test_data)

        batches = gather_inputs(
            test_data=test_data,
            forecasts=forecasts,
            quantile_levels=quantile_levels,
        )

        # only NumPy arrays are used from here on

        for metric_evaluator in self.metric_evaluators.values():
            metric_evaluator.reset()

        b = 0
        for data in batches:
            if b == 4:
                print("UH")
            print(f"batch {b}")
            b += 1
            for metric_evaluator in self.metric_evaluators.values():
                metric_evaluator.update(data)

        result = dict()
        for metric_name, metric_evaluator in self.metric_evaluators.items():
            result[metric_name] = metric_evaluator.get()
        return result
