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

import functools

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# First-party imports
from gluonts.evaluation import (
    Evaluator,
    MultivariateEvaluator,
    get_seasonality,
)
from gluonts.model.forecast import QuantileForecast, SampleForecast

from _data import M4, TIMESERIES

def assert_metric(metric, calc, exp):
    if not np.allclose(calc, exp):
        raise AssertionError(
            f"Scores for the metric {metric} do not match:\n"
            f"Expected: {exp}\n"
            f"Optained: {calc}"
        )


def naive_forecaster(ts, prediction_length, num_samples=100, target_dim=0):
    """
    :param ts: pandas.Series
    :param prediction_length:
    :param num_samples: number of sample paths
    :param target_dim: number of axes of target (0: scalar, 1: array, ...)
    :return: np.array with dimension (num_samples, prediction_length)
    """

    # naive prediction: last observed value
    naive_pred = ts.values[-prediction_length - 1]
    assert len(naive_pred.shape) == target_dim
    return np.tile(
        naive_pred,
        (num_samples, prediction_length) + tuple(1 for _ in range(target_dim)),
    )


def naive_multivariate_forecaster(ts, prediction_length, num_samples=100):
    return naive_forecaster(ts, prediction_length, num_samples, target_dim=1)


def calculate_metrics(
    timeseries,
    evaluator,
    ts_datastructure,
    has_nans=False,
    forecaster=naive_forecaster,
    input_type=iter,
    prediction_length=3,
    freq="1D",
):
    if has_nans:
        timeseries[0, 1] = np.nan
        timeseries[0, 7] = np.nan

    index = pd.date_range(
        pd.Timestamp("2018-1-1 01:00:00"),
        periods=timeseries.shape[1],
        freq="1D",
    )
    forecast_start = index[-prediction_length]

    predict = functools.partial(
        forecaster, prediction_length=prediction_length, num_samples=100
    )
    SampleForecast_ = functools.partial(
        SampleForecast, start_date=forecast_start, freq=freq
    )

    true_values = [ts_datastructure(ts, index=index) for ts in timeseries]
    predictions = map(SampleForecast_, map(predict, true_values))
    return evaluator(input_type(true_values), predictions)


@pytest.mark.parametrize("entry", M4)
def test_MASE_sMAPE_M4(entry):
    target = np.array(entry["data"])
    expected = entry["metrics"]

    agg_df, item_df = calculate_metrics(target, Evaluator(), pd.Series)

    assert_metric("MASE", agg_df["MASE"], expected["MASE"])
    assert_metric("sMAPE", agg_df["sMAPE"], expected["sMAPE"])
    assert_metric(
        "seasonal_error",
        item_df["seasonal_error"].values,
        expected["seasonal_error"],
    )


@pytest.mark.parametrize("entry", TIMESERIES)
def test_metrics(entry):
    agg_metrics, item_metrics = calculate_metrics(
        entry["data"],
        Evaluator(),
        pd.Series,
        has_nans=entry["apply_nans"],
        input_type=entry["input_type"],
    )

    for metric, score in agg_metrics.items():
        if metric in entry["metrics"]:
            assert_metric(metric, score, entry["metrics"][metric])


TIMESERIES_MULTIVARIATE = [
    np.ones((5, 10, 2), dtype=np.float64),
    np.ones((5, 10, 2), dtype=np.float64),
    np.ones((5, 10, 2), dtype=np.float64),
    np.stack(
        (
            np.arange(0, 50, dtype=np.float64).reshape(5, 10),
            np.arange(50, 100, dtype=np.float64).reshape(5, 10),
        ),
        axis=2,
    ),
    np.stack(
        (
            np.arange(0, 50, dtype=np.float64).reshape(5, 10),
            np.arange(50, 100, dtype=np.float64).reshape(5, 10),
        ),
        axis=2,
    ),
    np.stack(
        (
            np.arange(0, 50, dtype=np.float64).reshape(5, 10),
            np.arange(50, 100, dtype=np.float64).reshape(5, 10),
        ),
        axis=2,
    ),
]

RES_MULTIVARIATE = [
    {
        "MSE": 0.0,
        "0_MSE": 0.0,
        "1_MSE": 0.0,
        "abs_error": 0.0,
        "abs_target_sum": 15.0,
        "abs_target_mean": 1.0,
        "seasonal_error": 0.0,
        "MASE": 0.0,
        "sMAPE": 0.0,
        "MSIS": 0.0,
        "RMSE": 0.0,
        "NRMSE": 0.0,
        "ND": 0.0,
        "MAE_Coverage": 0.5,
        "m_sum_MSE": 0.0,
    },
    {
        "MSE": 0.0,
        "abs_error": 0.0,
        "abs_target_sum": 15.0,
        "abs_target_mean": 1.0,
        "seasonal_error": 0.0,
        "MASE": 0.0,
        "sMAPE": 0.0,
        "MSIS": 0.0,
        "RMSE": 0.0,
        "NRMSE": 0.0,
        "ND": 0.0,
        "MAE_Coverage": 0.5,
        "m_sum_MSE": 0.0,
    },
    {
        "MSE": 0.0,
        "abs_error": 0.0,
        "abs_target_sum": 30.0,
        "abs_target_mean": 1.0,
        "seasonal_error": 0.0,
        "MASE": 0.0,
        "sMAPE": 0.0,
        "MSIS": 0.0,
        "RMSE": 0.0,
        "NRMSE": 0.0,
        "ND": 0.0,
        "MAE_Coverage": 0.5,
        "m_sum_MSE": 0.0,
    },
    {
        "MSE": 4.666_666_666_666,
        "abs_error": 30.0,
        "abs_target_sum": 420.0,
        "abs_target_mean": 28.0,
        "seasonal_error": 1.0,
        "MASE": 2.0,
        "sMAPE": 0.113_254_049_3,
        "MSIS": 80.0,
        "RMSE": 2.160_246_899_469_286_9,
        "NRMSE": 0.077_151_674_981_045_956,
        "ND": 0.071_428_571_428_571_42,
        "MAE_Coverage": 0.5,
        "m_sum_MSE": 18.666_666_666_666,
    },
    {
        "MSE": 4.666_666_666_666,
        "abs_error": 30.0,
        "abs_target_sum": 1170.0,
        "abs_target_mean": 78.0,
        "seasonal_error": 1.0,
        "MASE": 2.0,
        "sMAPE": 0.026_842_301_756_499_45,
        "MSIS": 80.0,
        "RMSE": 2.160_246_899_469_286_9,
        "NRMSE": 0.027_695_473_070_119_065,
        "ND": 0.025_641_025_641_025_64,
        "MAE_Coverage": 0.5,
        "m_sum_MSE": 18.666_666_666_666,
    },
    {
        "MSE": 4.666_666_666_666,
        "abs_error": 60.0,
        "abs_target_sum": 1590.0,
        "abs_target_mean": 53.0,
        "seasonal_error": 1.0,
        "MASE": 2.0,
        "sMAPE": 0.070_048_175_528_249_73,
        "MSIS": 80.0,
        "RMSE": 2.160_246_899_469_286_9,
        "NRMSE": 0.040_759_375_461_684_65,
        "ND": 0.037_735_849_056_603_77,
        "MAE_Coverage": 0.5,
        "m_sum_MSE": 18.666_666_666_666,
    },
]

HAS_NANS_MULTIVARIATE = [False, False, False, False, False, False]

EVAL_DIMS = [[0], [1], [0, 1], [0], [1], None]

INPUT_TYPE = [list, list, iter, iter, list, iter]


@pytest.mark.parametrize(
    "timeseries, res, has_nans, eval_dims, input_type",
    zip(
        TIMESERIES_MULTIVARIATE,
        RES_MULTIVARIATE,
        HAS_NANS_MULTIVARIATE,
        EVAL_DIMS,
        INPUT_TYPE,
    ),
)
def test_metrics_multivariate(
    timeseries, res, has_nans, eval_dims, input_type
):
    evaluator = MultivariateEvaluator(
        eval_dims=eval_dims, target_agg_funcs={"sum": np.sum},
    )

    agg_metrics, item_metrics = calculate_metrics(
        timeseries,
        evaluator,
        pd.DataFrame,
        has_nans=has_nans,
        forecaster=naive_multivariate_forecaster,
        input_type=input_type,
    )

    for metric, score in agg_metrics.items():
        if metric in res.keys():
            assert abs(score - res[metric]) < 0.001, (
                "Scores for the metric {} do not match: \nexpected: {} "
                "\nobtained: {}".format(metric, res[metric], score)
            )


def test_evaluation_with_QuantileForecast():
    start = "2012-01-01"
    target = [2.4, 1.0, 3.0, 4.4, 5.5, 4.9] * 10
    index = pd.date_range(start=start, freq="1D", periods=len(target))
    ts = pd.Series(index=index, data=target)

    fcst = [
        QuantileForecast(
            start_date=pd.Timestamp("2012-01-01"),
            freq="D",
            forecast_arrays=np.array([[2.4, 9.0, 3.0, 2.4, 5.5, 4.9] * 10]),
            forecast_keys=["0.5"],
        )
    ]

    agg_metric, _ = Evaluator()([ts], fcst)

    assert np.isfinite(agg_metric["wQuantileLoss[0.5]"])


@pytest.mark.parametrize(
    "freq, expected_seasonality",
    [
        ("1H", 24),
        ("H", 24),
        ("2H", 12),
        ("3H", 8),
        ("4H", 6),
        ("15H", 1),
        ("5B", 1),
        ("1B", 5),
        ("2W", 1),
        ("3M", 4),
        ("1D", 1),
        ("7D", 1),
        ("8D", 1),
    ],
)
def test_get_seasonality(freq, expected_seasonality):
    assert get_seasonality(freq) == expected_seasonality
