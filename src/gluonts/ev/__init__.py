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

from .ts_stats import seasonal_error
from .stats import (
    absolute_label,
    error,
    absolute_error,
    squared_error,
    quantile_loss,
    coverage,
    absolute_percentage_error,
    symmetric_absolute_percentage_error,
    scaled_interval_score,
    absolute_scaled_error,
    scaled_quantile_loss,
    num_masked_target_values,
)
from .metrics import (
    Metric,
    MetricDefinition,
    MeanAbsoluteLabel,
    mean_absolute_label,
    SumAbsoluteLabel,
    sum_absolute_label,
    SumAbsoluteError,
    sum_absolute_error,
    MSE,
    mse,
    SumQuantileLoss,
    Coverage,
    MAPE,
    mape,
    SMAPE,
    smape,
    MSIS,
    msis,
    MASE,
    mase,
    ND,
    nd,
    RMSE,
    rmse,
    NRMSE,
    nrmse,
    WeightedSumQuantileLoss,
    MAECoverage,
    MeanScaledQuantileLoss,
    AverageMeanScaledQuantileLoss,
    MeanSumQuantileLoss,
    MeanWeightedSumQuantileLoss,
    OWA,
    owa,
    SumNumMaskedTargetValues,
)
from .aggregations import Aggregation, Sum, Mean


__all__ = [
    "seasonal_error",
    "absolute_label",
    "error",
    "absolute_error",
    "squared_error",
    "quantile_loss",
    "coverage",
    "absolute_percentage_error",
    "symmetric_absolute_percentage_error",
    "scaled_interval_score",
    "absolute_scaled_error",
    "scaled_quantile_loss",
    "Metric",
    "MetricDefinition",
    "MeanAbsoluteLabel",
    "mean_absolute_label",
    "SumAbsoluteLabel",
    "sum_absolute_label",
    "SumAbsoluteError",
    "sum_absolute_error",
    "MSE",
    "mse",
    "SumQuantileLoss",
    "Coverage",
    "MAPE",
    "mape",
    "SMAPE",
    "smape",
    "MSIS",
    "msis",
    "MASE",
    "mase",
    "ND",
    "nd",
    "RMSE",
    "rmse",
    "NRMSE",
    "nrmse",
    "WeightedSumQuantileLoss",
    "MAECoverage",
    "MeanScaledQuantileLoss",
    "AverageMeanScaledQuantileLoss",
    "MeanSumQuantileLoss",
    "MeanWeightedSumQuantileLoss",
    "OWA",
    "owa",
    "Aggregation",
    "Sum",
    "Mean",
    "SumNumMaskedTargetValues",
    "num_masked_target_values"
]


def evaluate(metrics, data_batches, axis=None):
    return metrics(axis).update_all(data_batches).get()
