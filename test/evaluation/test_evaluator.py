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

QUANTILES = [str(q / 10.0) for q in range(1, 10)]


def data_iterator(ts):
    """
    :param ts: list of pd.Series or pd.DataFrame
    :return:
    """
    for i in range(len(ts)):
        yield ts[i]


def fcst_iterator(fcst, start_dates, freq):
    """
    :param fcst: list of numpy arrays with the sample paths
    :return:
    """
    for i in range(len(fcst)):
        yield SampleForecast(
            samples=fcst[i], start_date=start_dates[i], freq=freq
        )


def iterator(it):
    """
    Convenience function to toggle whether to consume dataset and forecasts as iterators or iterables.
    :param it:
    :return: it (as iterator)
    """
    return iter(it)


def iterable(it):
    """
    Convenience function to toggle whether to consume dataset and forecasts as iterators or iterables.
    :param it:
    :return: it (as iterable)
    """
    return list(it)


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
    input_type=iterator,
):
    num_timeseries = timeseries.shape[0]
    num_timestamps = timeseries.shape[1]

    if has_nans:
        timeseries[0, 1] = np.nan
        timeseries[0, 7] = np.nan

    num_samples = 100
    prediction_length = 3
    freq = "1D"

    ts_start_dates = (
        []
    )  # starting date of each time series - can be different in general
    pd_timeseries = []  # list of pandas.DataFrame
    samples = []  # list of forecast samples
    start_dates = []  # start date of the prediction range
    for i in range(num_timeseries):
        ts_start_dates.append(pd.Timestamp(year=2018, month=1, day=1, hour=1))
        index = pd.date_range(
            ts_start_dates[i], periods=num_timestamps, freq=freq
        )

        pd_timeseries.append(ts_datastructure(timeseries[i], index=index))
        samples.append(
            forecaster(pd_timeseries[i], prediction_length, num_samples)
        )
        start_dates.append(
            pd.date_range(
                ts_start_dates[i], periods=num_timestamps, freq=freq
            )[-prediction_length]
        )

    # data iterator
    data_iter = input_type(data_iterator(pd_timeseries))
    fcst_iter = input_type(fcst_iterator(samples, start_dates, freq))

    # evaluate
    agg_df, item_df = evaluator(data_iter, fcst_iter)
    return agg_df, item_df


TIMESERIES_M4 = [
    np.array(
        [
            [
                2.943_013,
                2.822_251,
                4.196_222,
                1.328_664,
                4.947_390,
                3.333_131,
                1.479_800,
                2.265_094,
                3.413_493,
                3.497_607,
            ],
            [
                -0.126_781_2,
                3.057_412_2,
                1.901_594_4,
                2.772_549_5,
                3.312_853_1,
                4.411_818_0,
                3.709_025_2,
                4.322_028,
                2.565_359,
                3.074_308,
            ],
            [
                2.542_998,
                2.336_757,
                1.417_916,
                1.335_139,
                2.523_035,
                3.645_589,
                3.382_819,
                2.075_960,
                2.643_869,
                2.772_456,
            ],
            [
                0.315_685_6,
                1.892_312_1,
                2.476_861_2,
                3.511_628_6,
                4.384_346_5,
                2.960_685_6,
                4.897_572_5,
                3.280_125,
                4.768_556,
                4.958_616,
            ],
            [
                2.205_877_3,
                0.782_759_4,
                2.401_420_8,
                2.385_643_4,
                4.845_818_2,
                3.102_322_9,
                3.567_723_7,
                4.878_143,
                3.735_245,
                2.218_113,
            ],
        ]
    ),
    np.array(
        [
            [
                13.11301,
                13.16225,
                14.70622,
                12.00866,
                15.79739,
                14.35313,
                12.66980,
                13.62509,
                14.94349,
                15.19761,
            ],
            [
                10.04322,
                13.39741,
                12.41159,
                13.45255,
                14.16285,
                15.43182,
                14.89903,
                15.68203,
                14.09536,
                14.77431,
            ],
            [
                12.71300,
                12.67676,
                11.92792,
                12.01514,
                13.37303,
                14.66559,
                14.57282,
                13.43596,
                14.17387,
                14.47246,
            ],
            [
                10.48569,
                12.23231,
                12.98686,
                14.19163,
                15.23435,
                13.98069,
                16.08757,
                14.64012,
                16.29856,
                16.65862,
            ],
            [
                12.37588,
                11.12276,
                12.91142,
                13.06564,
                15.69582,
                14.12232,
                14.75772,
                16.23814,
                15.26524,
                13.91811,
            ],
        ]
    ),
]

RES_M4 = [
    {
        "MASE": 0.816_837_618,
        "MAPE": 0.324_517_430_685_928_1,
        "sMAPE": 0.326_973_268_4,
        "seasonal_error": np.array(
            [1.908_101, 1.258_838, 0.63018, 1.238_201, 1.287_771]
        ),
    },
    {
        "MASE": 0.723_948_2,
        "MAPE": 0.063_634_129_851_747_6,
        "sMAPE": 0.065_310_85,
        "seasonal_error": np.array(
            [1.867_847, 1.315_505, 0.602_587_4, 1.351_535, 1.339_179]
        ),
    },
]


@pytest.mark.parametrize("timeseries, res", zip(TIMESERIES_M4, RES_M4))
def test_MASE_sMAPE_M4(timeseries, res):
    ts_datastructure = pd.Series
    evaluator = Evaluator(quantiles=QUANTILES)
    agg_df, item_df = calculate_metrics(
        timeseries, evaluator, ts_datastructure
    )

    assert abs((agg_df["MASE"] - res["MASE"]) / res["MASE"]) < 0.001, (
        "Scores for the metric MASE do not match: "
        "\nexpected: {} \nobtained: {}".format(res["MASE"], agg_df["MASE"])
    )
    assert abs((agg_df["MAPE"] - res["MAPE"]) / res["MAPE"]) < 0.001, (
        "Scores for the metric MAPE do not match: \nexpected: {} "
        "\nobtained: {}".format(res["MAPE"], agg_df["MAPE"])
    )
    assert abs((agg_df["sMAPE"] - res["sMAPE"]) / res["sMAPE"]) < 0.001, (
        "Scores for the metric sMAPE do not match: \nexpected: {} "
        "\nobtained: {}".format(res["sMAPE"], agg_df["sMAPE"])
    )
    assert (
        sum(abs(item_df["seasonal_error"].values - res["seasonal_error"]))
        < 0.001
    ), (
        "Scores for the metric seasonal_error do not match: \nexpected: {} "
        "\nobtained: {}".format(
            res["seasonal_error"], item_df["seasonal_error"].values
        )
    )


TIMESERIES = [
    np.ones((5, 10), dtype=np.float64),
    np.ones((5, 10), dtype=np.float64),
    np.arange(0, 50, dtype=np.float64).reshape(5, 10),
    np.arange(0, 50, dtype=np.float64).reshape(5, 10),
    np.array([[np.nan] * 10, [1.0] * 10]),
]

RES = [
    {
        "MSE": 0.0,
        "abs_error": 0.0,
        "abs_target_sum": 15.0,
        "abs_target_mean": 1.0,
        "seasonal_error": 0.0,
        "MASE": 0.0,
        "MAPE": 0.0,
        "sMAPE": 0.0,
        "MSIS": 0.0,
        "RMSE": 0.0,
        "NRMSE": 0.0,
        "ND": 0.0,
        "MAE_Coverage": 0.5,
    },
    {
        "MSE": 0.0,
        "abs_error": 0.0,
        "abs_target_sum": 14.0,
        "abs_target_mean": 1.0,
        "seasonal_error": 0.0,
        "MASE": 0.0,
        "MAPE": 0.0,
        "sMAPE": 0.0,
        "MSIS": 0.0,
        "RMSE": 0.0,
        "NRMSE": 0.0,
        "ND": 0.0,
        "MAE_Coverage": 0.5,
    },
    {
        "MSE": 4.666_666_666_666,
        "abs_error": 30.0,
        "abs_target_sum": 420.0,
        "abs_target_mean": 28.0,
        "seasonal_error": 1.0,
        "MASE": 2.0,
        "MAPE": 0.103_112_211_532_524_85,
        "sMAPE": 0.113_254_049_3,
        "MSIS": 80.0,
        "RMSE": 2.160_246_899_469_286_9,
        "NRMSE": 0.077_151_674_981_045_956,
        "ND": 0.071_428_571_428_571_42,
        "MAE_Coverage": 0.5,
    },
    {
        "MSE": 5.033_333_333_333_3,
        "abs_error": 29.0,
        "abs_target_sum": 413.0,
        "abs_target_mean": 28.1,
        "seasonal_error": 1.0,
        "MASE": 2.1,
        "MAPE": 0.113_032_846_453_159_77,
        "sMAPE": 0.125_854_781_903_299_57,
        "MSIS": 84.0,
        "RMSE": 2.243_509_156_061_845_6,
        "NRMSE": 0.079_840_183_489_745_39,
        "ND": 0.070_217_917_675_544_79,
        "MAE_Coverage": 0.5,
    },
    {
        "MSE": 0.0,
        "abs_error": 0.0,
        "abs_target_sum": 3.0,
        "abs_target_mean": 1.0,
        "seasonal_error": 0.0,
        "MASE": 0.0,
        "MAPE": 0.0,
        "sMAPE": 0.0,
        "MSIS": 0.0,
        "RMSE": 0.0,
        "NRMSE": 0.0,
        "ND": 0.0,
        "MAE_Coverage": 0.5,
    },
]

HAS_NANS = [False, True, False, True, True]


INPUT_TYPE = [iterable, iterable, iterator, iterator, iterable]


@pytest.mark.parametrize(
    "timeseries, res, has_nans, input_type",
    zip(TIMESERIES, RES, HAS_NANS, INPUT_TYPE),
)
def test_metrics(timeseries, res, has_nans, input_type):
    ts_datastructure = pd.Series
    evaluator = Evaluator(quantiles=QUANTILES, num_workers=0)
    agg_metrics, item_metrics = calculate_metrics(
        timeseries,
        evaluator,
        ts_datastructure,
        has_nans=has_nans,
        input_type=input_type,
    )

    for metric, score in agg_metrics.items():
        if metric in res.keys():
            assert abs(score - res[metric]) < 0.001, (
                "Scores for the metric {} do not match: \nexpected: {} "
                "\nobtained: {}".format(metric, res[metric], score)
            )


@pytest.mark.parametrize(
    "timeseries, res, has_nans, input_type",
    zip(TIMESERIES, RES, HAS_NANS, INPUT_TYPE),
)
def test_metrics_mp(timeseries, res, has_nans, input_type):
    ts_datastructure = pd.Series
    # Default will be multiprocessing evaluator
    evaluator = Evaluator(quantiles=QUANTILES, num_workers=4)
    agg_metrics, item_metrics = calculate_metrics(
        timeseries,
        evaluator,
        ts_datastructure,
        has_nans=has_nans,
        input_type=input_type,
    )

    for metric, score in agg_metrics.items():
        if metric in res.keys():
            assert abs(score - res[metric]) < 0.001, (
                "Scores for the metric {} do not match: \nexpected: {} "
                "\nobtained: {}".format(metric, res[metric], score)
            )


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

INPUT_TYPE = [iterable, iterable, iterator, iterator, iterable, iterator]


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
    ts_datastructure = pd.DataFrame
    evaluator = MultivariateEvaluator(
        quantiles=QUANTILES,
        eval_dims=eval_dims,
        target_agg_funcs={"sum": np.sum},
    )

    agg_metrics, item_metrics = calculate_metrics(
        timeseries,
        evaluator,
        ts_datastructure,
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
    start = "2012-01-11"
    target = [2.4, 1.0, 3.0, 4.4, 5.5, 4.9] * 11
    index = pd.date_range(start=start, freq="1D", periods=len(target))
    ts = pd.Series(index=index, data=target)

    ev = Evaluator(quantiles=("0.1", "0.2", "0.5"))

    fcst = [
        QuantileForecast(
            start_date=pd.Timestamp("2012-01-11"),
            freq="D",
            forecast_arrays=np.array([[2.4, 9.0, 3.0, 2.4, 5.5, 4.9] * 10]),
            forecast_keys=["0.5"],
        )
    ]

    agg_metric, _ = ev(iter([ts]), iter(fcst))

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
