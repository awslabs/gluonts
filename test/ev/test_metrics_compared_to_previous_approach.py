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

import numpy as np

from toolz import take

from gluonts.dataset.split import TestData, TestTemplate, OffsetSplitter
from gluonts.model.forecast import Quantile
from gluonts.model.predictor import Predictor
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

from gluonts.ev.metrics import (
    MAPE,
    MASE,
    MSE,
    MSIS,
    SMAPE,
    Coverage,
    SumAbsoluteError,
    SumQuantileLoss,
    mean_absolute_label,
    sum_absolute_label,
)


def get_old_metrics(dataset, predictor):
    forecast_it, ts_it = make_evaluation_predictions(dataset, predictor)
    evaluator = Evaluator(quantiles=(0.1, 0.5, 0.9))
    agg_metrics, item_metrics = evaluator(ts_it, forecast_it)
    return agg_metrics


def get_new_metrics(test_data: TestData, predictor: Predictor):
    """simulate former Evaluator class by first aggregating per time series
    and then aggregating once more"""
    quantiles = [Quantile.parse(q) for q in (0.1, 0.5, 0.9)]

    metrics_to_evaluate = [
        sum_absolute_label,
        SumAbsoluteError(),
        *(SumQuantileLoss(q=quantile.value) for quantile in quantiles),
        mean_absolute_label,
        MSE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        *(Coverage(q=quantile.value) for quantile in quantiles),
    ]

    item_metrics = predictor.backtest(
        test_data=test_data, metrics=metrics_to_evaluate, axis=1
    )

    aggregated_metrics = {
        "abs_target_mean": np.nanmean(item_metrics["mean_absolute_label"]),
        "MSE": np.nanmean(item_metrics["MSE"]),
        "MASE": np.nanmean(item_metrics["MASE"]),
        "MAPE": np.nanmean(item_metrics["MAPE"]),
        "sMAPE": np.nanmean(item_metrics["sMAPE"]),
        "MSIS": np.nanmean(item_metrics["MSIS"]),
        **{
            quantile.coverage_name: np.nanmean(
                item_metrics[f"coverage[{quantile.value}]"]
            )
            for quantile in quantiles
        },
        "abs_error": np.nansum(item_metrics["sum_absolute_error"]),
        "abs_target_sum": np.nansum(item_metrics["sum_absolute_label"]),
        **{
            quantile.loss_name: np.nansum(
                item_metrics[f"sum_quantile_loss[{quantile.value}]"]
            )
            for quantile in quantiles
        },
    }

    aggregated_metrics["RMSE"] = np.sqrt(aggregated_metrics["MSE"])
    aggregated_metrics["NRMSE"] = (
        aggregated_metrics["RMSE"] / aggregated_metrics["abs_target_mean"]
    )
    aggregated_metrics["ND"] = (
        aggregated_metrics["abs_error"] / aggregated_metrics["abs_target_sum"]
    )
    for quantile in quantiles:
        aggregated_metrics[quantile.weighted_loss_name] = (
            aggregated_metrics[quantile.loss_name]
            / aggregated_metrics["abs_target_sum"]
        )

    aggregated_metrics["mean_absolute_QuantileLoss"] = np.array(
        [aggregated_metrics[quantile.loss_name] for quantile in quantiles]
    ).mean()

    aggregated_metrics["mean_wQuantileLoss"] = np.array(
        [
            aggregated_metrics[quantile.weighted_loss_name]
            for quantile in quantiles
        ]
    ).mean()

    aggregated_metrics["MAE_Coverage"] = np.mean(
        [
            np.abs(
                aggregated_metrics[quantile.coverage_name]
                - np.array([quantile.value])
            )
            for quantile in quantiles
        ]
    )

    return aggregated_metrics


def test_against_former_evaluator():
    dataset = get_dataset("electricity")

    prediction_length = dataset.metadata.prediction_length
    freq = dataset.metadata.freq

    dataset_test = list(take(1000, dataset.test))

    test_template = TestTemplate(
        dataset=dataset_test,
        splitter=OffsetSplitter(offset=-prediction_length),
    )

    test_data = test_template.generate_instances(
        prediction_length=prediction_length
    )

    predictor = SeasonalNaivePredictor(
        prediction_length=prediction_length, freq=freq
    )
    evaluation_result = get_old_metrics(dataset_test, predictor)
    ev_result = get_new_metrics(test_data, predictor)

    for metric_name in ev_result.keys():
        # low precision of decimal=1 is used because masked arrays (which the
        # old implementation uses) have their own implementation of mean but
        # not nanmean (which is used in Mean aggregation)
        # - see https://github.com/numpy/numpy/issues/9071

        if ev_result[metric_name] == np.inf:
            continue  # TODO: handle inf values

        np.testing.assert_almost_equal(
            ev_result[metric_name], evaluation_result[metric_name], decimal=1
        )
