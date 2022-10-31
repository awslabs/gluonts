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
import pytest

from gluonts.ev.metrics import (
    absolute_error,
    absolute_label,
    absolute_percentage_error,
    coverage,
    error,
    quantile_loss,
    squared_error,
    symmetric_absolute_percentage_error,
    scaled_interval_score,
    absolute_scaled_error,
)
from gluonts.ev.ts_stats import seasonal_error

PREDICTION_LENGTH = 5

NAN = np.full(PREDICTION_LENGTH, np.nan)
ZEROES = np.zeros(PREDICTION_LENGTH)
CONSTANT = np.full(PREDICTION_LENGTH, 0.4)
LINEAR = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
EXPONENTIAL = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001])


@pytest.mark.parametrize(
    "label, forecast, stats",
    [
        (
            ZEROES,
            ZEROES,
            [
                (absolute_label, {}, ZEROES),
                (error, {"forecast_type": "mean"}, ZEROES),
                (absolute_error, {"forecast_type": "mean"}, ZEROES),
                (squared_error, {"forecast_type": "mean"}, ZEROES),
                (quantile_loss, {"q": 0.5}, ZEROES),
                (coverage, {"q": 0.5}, ZEROES),
                (absolute_percentage_error, {"forecast_type": "mean"}, NAN),
                (
                    symmetric_absolute_percentage_error,
                    {"forecast_type": "mean"},
                    NAN,
                ),
            ],
        ),
        (
            LINEAR,
            ZEROES,
            [
                (absolute_label, {}, LINEAR),
                (error, {"forecast_type": "mean"}, LINEAR),
                (absolute_error, {"forecast_type": "mean"}, LINEAR),
                (
                    squared_error,
                    {"forecast_type": "mean"},
                    np.array([0.0, 0.01, 0.04, 0.09, 0.16]),
                ),
                (quantile_loss, {"q": 0.5}, LINEAR),
                (coverage, {"q": 0.5}, ZEROES),
                (
                    absolute_percentage_error,
                    {"forecast_type": "mean"},
                    np.array([np.nan, 1.0, 1.0, 1.0, 1.0]),
                ),
                (
                    symmetric_absolute_percentage_error,
                    {"forecast_type": "mean"},
                    np.array([np.nan, 2.0, 2.0, 2.0, 2.0]),
                ),
            ],
        ),
        (
            LINEAR,
            CONSTANT,
            [
                (absolute_label, {}, LINEAR),
                (
                    error,
                    {"forecast_type": "mean"},
                    np.array([-0.4, -0.3, -0.2, -0.1, 0.0]),
                ),
                (
                    absolute_error,
                    {"forecast_type": "mean"},
                    np.array([0.4, 0.3, 0.2, 0.1, 0.0]),
                ),
                (
                    squared_error,
                    {"forecast_type": "mean"},
                    np.array([0.16, 0.09, 0.04, 0.01, 0.0]),
                ),
                (
                    quantile_loss,
                    {"q": 0.5},
                    np.array([0.4, 0.3, 0.2, 0.1, 0.0]),
                ),
                (coverage, {"q": 0.5}, np.array([1.0, 1.0, 1.0, 1.0, 0.0])),
                (
                    absolute_percentage_error,
                    {"forecast_type": "mean"},
                    np.array([np.inf, 3.0, 1.0, 0.333_333_333_333, 0.0]),
                ),
                (
                    symmetric_absolute_percentage_error,
                    {"forecast_type": "mean"},
                    np.array(
                        [2.0, 1.2, 0.666_666_666_666, 0.285_714_285_714, 0.0]
                    ),
                ),
            ],
        ),
    ],
)
def test_stats_without_seasonal_error(label, forecast, stats):
    data = {
        "label": label,
        "0.5": forecast,
        "mean": forecast,
    }

    for stat, kwargs, expected_result in stats:
        np.testing.assert_almost_equal(stat(data, **kwargs), expected_result)


@pytest.mark.parametrize(
    "label, forecast, stats",
    [
        (
            ZEROES,
            ZEROES,
            [
                (scaled_interval_score, {"alpha": 0.05}, NAN),
                (absolute_scaled_error, {"forecast_type": "0.975"}, NAN),
            ],
        ),
        (
            LINEAR,
            ZEROES,
            [
                (
                    scaled_interval_score,
                    {"alpha": 0.05},
                    np.array([0.0, 20.0, 40.0, 60.0, 80.0]),
                ),
                (
                    absolute_scaled_error,
                    {"forecast_type": "0.975"},
                    np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
                ),
            ],
        ),
        (
            LINEAR,
            CONSTANT,
            [
                (
                    scaled_interval_score,
                    {"alpha": 0.05},
                    np.array([3.9, 1.9, 1.9, 1.9, 3.9]),
                ),
                (
                    absolute_scaled_error,
                    {"forecast_type": "0.975"},
                    np.array([1.95, 1.45, 0.95, 0.45, 0.05]),
                ),
            ],
        ),
    ],
)
def test_metrics_with_seasonal_error(label, forecast, stats):
    data = {
        "label": label,
        "mean": forecast,
        # applying `seasonal_error` on the label doesn't make much sense but
        # at least, the function gets used and breaking changes are detected
        "seasonal_error": seasonal_error(label, seasonality=2),
        # to keep things simple, the following values are very arbitrary, too
        "0.025": forecast * 0.025,
        "0.975": forecast * 0.975,
    }

    for stat, kwargs, expected_result in stats:
        np.testing.assert_almost_equal(stat(data, **kwargs), expected_result)


# TODO: Consider using this alternative way to test stats.
# It feels like cheating because we're calculating pretty much the same things
# as in the functions but in fact is more thorough than above tests because
# all combinations are compared.

RANDOM = np.random.random(PREDICTION_LENGTH)
NEGATIVE_RANDOM = -1 * np.random.random(PREDICTION_LENGTH)
VALUES = [NAN, ZEROES, CONSTANT, LINEAR, EXPONENTIAL, RANDOM, NEGATIVE_RANDOM]


def test_absolute_label():
    for label in VALUES:
        for forecast in VALUES:
            data = {"label": label, "mean": forecast}
            np.testing.assert_almost_equal(absolute_label(data), np.abs(label))


def test_error():
    for label in VALUES:
        for forecast in VALUES:
            data = {"label": label, "mean": forecast}
            np.testing.assert_almost_equal(
                error(data, forecast_type="mean"), label - forecast
            )
