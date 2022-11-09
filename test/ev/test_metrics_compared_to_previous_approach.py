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

from copy import deepcopy
from typing import Dict, Tuple

import numpy as np

from gluonts.dataset import Dataset
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.split import TestData, TestTemplate, OffsetSplitter
from gluonts.model.forecast import Quantile
from gluonts.model.predictor import Predictor, get_backtest_input
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.ev import (
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


def get_old_metrics(
    dataset: Dataset, predictor: Predictor, quantile_levels: Tuple[float]
) -> Dict[str, np.ndarray]:
    forecast_it, ts_it = make_evaluation_predictions(dataset, predictor)
    evaluator = Evaluator(quantiles=quantile_levels)
    agg_metrics, item_metrics = evaluator(ts_it, forecast_it)

    return agg_metrics


def get_masked_test_data(test_data: TestData) -> TestData:
    masked_dataset = []
    for data_entry in test_data.dataset:
        masked_entry = deepcopy(data_entry)
        masked_entry["target"] = np.ma.masked_invalid(data_entry["target"])
        masked_dataset.append(masked_entry)

    masked_test_data = TestData(
        dataset=masked_dataset,
        splitter=test_data.splitter,
        prediction_length=test_data.prediction_length,
        windows=test_data.windows,
        distance=test_data.distance,
        max_history=test_data.max_history,
    )

    return masked_test_data


def get_new_metrics(
    test_data: TestData, predictor: Predictor, quantile_levels: Tuple[float]
) -> Dict[str, np.ndarray]:
    """Simulate former Evaluator by doing two-step aggregations."""

    quantiles = [Quantile.parse(q) for q in quantile_levels]

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

    masked_test_data = get_masked_test_data(test_data)
    item_metrics = predictor.backtest(
        test_data=masked_test_data, metrics=metrics_to_evaluate, axis=1
    )

    aggregated_metrics = {
        "abs_target_mean": np.ma.mean(item_metrics["mean_absolute_label"]),
        "MSE": np.ma.mean(item_metrics["MSE"]),
        "MASE": np.ma.mean(item_metrics["MASE"]),
        "MAPE": np.ma.mean(item_metrics["MAPE"]),
        "sMAPE": np.ma.mean(item_metrics["sMAPE"]),
        "MSIS": np.ma.mean(item_metrics["MSIS"]),
        **{
            quantile.coverage_name: np.ma.mean(
                item_metrics[f"coverage[{quantile.value}]"]
            )
            for quantile in quantiles
        },
        "abs_error": np.ma.sum(item_metrics["sum_absolute_error"]),
        "abs_target_sum": np.ma.sum(item_metrics["sum_absolute_label"]),
        **{
            quantile.loss_name: np.ma.sum(
                item_metrics[f"sum_quantile_loss[{quantile.value}]"]
            )
            for quantile in quantiles
        },
    }

    seasonal_errors = []
    forecasts = predictor.predict(dataset=masked_test_data.input)
    for data in get_backtest_input(masked_test_data, forecasts):
        seasonal_errors.append(data["seasonal_error"])

    aggregated_metrics["seasonal_error"] = np.ma.mean(
        np.stack(seasonal_errors)
    )

    # For the following metrics, the new implementations are **not** being
    # used, because they don't follow the two-step approach. Using the new
    # implementation with `axis=None` produces slightly different results.

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

    aggregated_metrics["MAE_Coverage"] = np.ma.mean(
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
    dataset = get_dataset("exchange_rate")

    prediction_length = dataset.metadata.prediction_length
    freq = dataset.metadata.freq

    test_template = TestTemplate(
        dataset=dataset.test,
        splitter=OffsetSplitter(offset=-prediction_length),
    )

    test_data = test_template.generate_instances(
        prediction_length=prediction_length
    )

    predictor = SeasonalNaivePredictor(
        prediction_length=prediction_length, freq=freq
    )

    quantile_levels = (0.1, 0.5, 0.9)
    evaluation_result = get_old_metrics(
        dataset.test, predictor, quantile_levels
    )
    ev_result = get_new_metrics(test_data, predictor, quantile_levels)

    for metric_name in ev_result.keys():
        # Using decimal=4 to account for some inprecisions that likely stem
        # from the fact that the old evaluation approach uses Pandas.
        # Taking the first 1000 entries of "electricity" for example works
        # fine even with the default precision of decimal=7.
        np.testing.assert_almost_equal(
            ev_result[metric_name],
            evaluation_result[metric_name],
            decimal=4,
        )
