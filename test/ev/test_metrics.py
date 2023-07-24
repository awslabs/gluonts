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

from typing import Optional

import numpy as np
import pytest

from gluonts.ev.metrics import (
    Coverage,
    MAECoverage,
    MetricDefinition,
    MSE,
    MAE,
    MAPE,
    SMAPE,
    MASE,
    MeanScaledQuantileLoss,
    AverageMeanScaledQuantileLoss,
    WeightedSumQuantileLoss,
    MeanWeightedSumQuantileLoss,
    ND,
    RMSE,
    NRMSE,
)
from gluonts.ev.ts_stats import seasonal_error


METRICS = [
    Coverage(0.5),
    MAECoverage([0.1, 0.5, 0.9]),
    MSE(),
    MAE(),
    MAPE(),
    SMAPE(),
    MASE(),
    WeightedSumQuantileLoss(0.5),
    MeanWeightedSumQuantileLoss([0.1, 0.5, 0.9]),
    MeanScaledQuantileLoss(0.5),
    AverageMeanScaledQuantileLoss([0.1, 0.5, 0.9]),
    ND(),
    RMSE(),
    NRMSE(),
]


@pytest.mark.parametrize(
    "metric",
    METRICS,
)
@pytest.mark.parametrize("axis", [None, (0, 1), (0,), (1,), ()])
def test_metric(metric: MetricDefinition, axis: Optional[tuple]):
    input_length = 20
    label_length = 5
    num_entries = 7

    data = [
        {
            "input": np.random.normal(size=(1, input_length)),
            "label": np.random.normal(size=(1, label_length)),
            "0.1": np.random.normal(size=(1, label_length)),
            "0.5": np.random.normal(size=(1, label_length)),
            "0.9": np.random.normal(size=(1, label_length)),
            "mean": np.random.normal(size=(1, label_length)),
        }
        for _ in range(num_entries)
    ]

    for entry in data:
        entry["seasonal_error"] = seasonal_error(
            entry["input"], seasonality=1, time_axis=1
        )

    metric_value = metric(axis=axis).update_all(data).get()

    if axis is None or axis == (0, 1):
        assert isinstance(metric_value, float)
    elif axis == (0,):
        assert isinstance(metric_value, np.ndarray)
        assert metric_value.shape == (label_length,)
    elif axis == (1,):
        assert isinstance(metric_value, np.ndarray)
        assert metric_value.shape == (num_entries,)
    elif axis == ():
        assert isinstance(metric_value, np.ndarray)
        assert metric_value.shape == (num_entries, label_length)
    else:
        raise ValueError("unsupported axis")

    return metric_value
