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

from pydantic.decorator import Callable
from gluonts.evaluation.metrics import (
    abs_target_mean,
    abs_target_sum,
    mase,
    mape,
    msis,
    owa,
    quantile_loss,
    smape,
    mse,
    abs_error,
    coverage,
)
import numpy as np
import pytest
import pandas as pd

ZEROES = np.array([0.0] * 5)
LINEAR = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
EXPONENTIAL = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001])
CONSTANT = np.array([0.4] * 5)


# TODO remove `"exclude_zero_denominator": True` when metrics are refactored
@pytest.mark.parametrize(
    "target, forecast, metrics",
    [
        (
            ZEROES,
            ZEROES,
            [
                (
                    mase,
                    np.nan,
                    {"seasonal_error": 0.0, "exclude_zero_denominator": False},
                ),
                (
                    mase,
                    0.0,
                    {"seasonal_error": 1.0, "exclude_zero_denominator": False},
                ),
                (mse, 0.0, {}),
                (abs_error, 0.0, {}),
                (quantile_loss, 0.0, {"q": 0.5}),
                (coverage, 0.0, {}),
                (mape, np.nan, {"exclude_zero_denominator": False}),
                (smape, np.nan, {"exclude_zero_denominator": False}),
                # TODO add OWA when exclude_zero_denominator is removed
            ],
        ),
        (
            LINEAR,
            ZEROES,
            [
                (
                    mase,
                    0.2 / 10e-10,
                    {
                        "seasonal_error": 10e-10,
                        "exclude_zero_denominator": False,
                    },
                ),
                (
                    mase,
                    np.inf,
                    {"seasonal_error": 0.0, "exclude_zero_denominator": False},
                ),
                (mse, 0.06, {}),
                (abs_error, 1.0, {}),
                (quantile_loss, 0.2, {"q": 0.1}),
                (quantile_loss, 1.0, {"q": 0.5}),
                (quantile_loss, 1.8, {"q": 0.9}),
                (coverage, 0.0, {}),
                (mape, np.nan, {"exclude_zero_denominator": False}),
                (smape, np.nan, {"exclude_zero_denominator": False}),
                (
                    owa,
                    1.7041284403669723,
                    {
                        "past_data": LINEAR,
                        "seasonal_error": 0.5,
                        "start_date": pd.Timestamp("2020-01-20", freq="H"),
                    },
                ),
            ],
        ),
        (
            LINEAR,
            CONSTANT,
            [
                (
                    mase,
                    0.2,
                    {"exclude_zero_denominator": False, "seasonal_error": 1},
                ),
                (mse, 0.06, {}),
                (abs_error, 1.0, {}),
                (quantile_loss, 1.8, {"q": 0.1}),
                (quantile_loss, 1.0, {"q": 0.5}),
                (quantile_loss, 0.2, {"q": 0.9}),
                (coverage, 0.8, {}),
                (mape, np.inf, {"exclude_zero_denominator": False}),
                (
                    smape,
                    0.8304761904761906,
                    {"exclude_zero_denominator": False},
                ),
                (
                    owa,
                    1.0,
                    {
                        "past_data": LINEAR,
                        "seasonal_error": 0.5,
                        "start_date": pd.Timestamp("2020-01-20", freq="H"),
                    },
                ),
            ],
        ),
        (
            ZEROES,
            EXPONENTIAL,
            [
                (
                    mase,
                    0.022222,
                    {"exclude_zero_denominator": False, "seasonal_error": 1},
                ),
                (mse, 0.00202020202, {}),
                (abs_error, 0.11111, {}),
                (quantile_loss, 0.199998, {"q": 0.1}),
                (quantile_loss, 0.11111, {"q": 0.5}),
                (quantile_loss, 0.0222219, {"q": 0.9}),
                (coverage, 1.0, {}),
                (mape, np.nan, {}),
                (mape, np.inf, {"exclude_zero_denominator": False}),
                (smape, 2.0, {}),
                (smape, 2.0, {"exclude_zero_denominator": False}),
                # TODO add OWA when exclude_zero_denominator is removed
            ],
        ),
        (
            np.array([1.0, 2.0, 3.0, 0.0, 0.0]),
            np.array([2.0, 3.0, 4.0, 1.0, 0.0]),
            [
                (
                    mase,
                    0.8,
                    {"exclude_zero_denominator": False, "seasonal_error": 1},
                ),
                (mse, 0.8, {}),
                (abs_error, 4.0, {}),
                (quantile_loss, 7.2, {"q": 0.1}),
                (quantile_loss, 4.0, {"q": 0.5}),
                (quantile_loss, 0.7999999, {"q": 0.9}),
                (coverage, 0.8, {}),
                (mape, 0.6111111, {}),
                (mape, np.nan, {"exclude_zero_denominator": False}),
                (smape, 0.8380952, {}),
                (smape, np.nan, {"exclude_zero_denominator": False}),
                (
                    owa,
                    0.6285506945884303,
                    {
                        "past_data": CONSTANT,
                        "seasonal_error": 0.5,
                        "start_date": pd.Timestamp("2020-01-20", freq="H"),
                    },
                ),
            ],
        ),
        (
            CONSTANT,
            EXPONENTIAL,
            [
                (
                    mase,
                    0.377778,
                    {"exclude_zero_denominator": False, "seasonal_error": 1},
                ),
                (mse, 0.14424260202, {}),
                (abs_error, 1.88889, {}),
                (quantile_loss, 0.377778, {"q": 0.1}),
                (quantile_loss, 1.88889, {"q": 0.5}),
                (quantile_loss, 3.400002, {"q": 0.9}),
                (coverage, 0.0, {}),
                (mape, 0.944445, {"exclude_zero_denominator": False}),
                (smape, 1.8182728, {"exclude_zero_denominator": False}),
                (
                    owa,
                    np.inf,
                    {
                        "past_data": CONSTANT,
                        "seasonal_error": 0.5,
                        "start_date": pd.Timestamp("2020-01-20", freq="H"),
                    },
                ),
            ],
        ),
    ],
)
def test_metrics(target, forecast, metrics):
    for metric, expected, kwargs in metrics:
        np.testing.assert_almost_equal(
            metric(target, forecast, **kwargs), expected
        )


@pytest.mark.parametrize(
    "target, metric, expected",
    [
        (ZEROES, abs_target_sum, 0.0),
        (LINEAR, abs_target_sum, 1.0),
        (ZEROES, abs_target_mean, 0.0),
        (LINEAR, abs_target_mean, 0.2),
    ],
)
def test_target_metrics(target, metric, expected):
    np.testing.assert_almost_equal(metric(target), expected)


@pytest.mark.parametrize(
    "target, lower_quantile, upper_quantile, seasonal_error, alpha, expected",
    [
        (LINEAR, ZEROES, CONSTANT, 0.0, 0.05, np.inf),
        (LINEAR, ZEROES, CONSTANT, 1.0, 0.05, 0.4),
        (LINEAR, ZEROES, CONSTANT, 0.01, 0.05, 40.0),
        (ZEROES, ZEROES, ZEROES, 0.0, 0.05, np.nan),
    ],
)
def test_msis(
    target, lower_quantile, upper_quantile, seasonal_error, alpha, expected
):
    np.testing.assert_almost_equal(
        msis(
            target=target,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            seasonal_error=seasonal_error,
            alpha=alpha,
            exclude_zero_denominator=False,  # TODO remove when refactoring msis
        ),
        expected,
    )
