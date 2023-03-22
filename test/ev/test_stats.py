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

from gluonts.ev import (
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
    seasonal_error,
)

PREDICTION_LENGTH = 5

NAN = np.full(PREDICTION_LENGTH, np.nan)
ZEROES = np.zeros(PREDICTION_LENGTH)
CONSTANT = np.full(PREDICTION_LENGTH, 0.4)
LINEAR = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
EXPONENTIAL = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001])
RANDOM = np.random.random(PREDICTION_LENGTH)
NEGATIVE_RANDOM = -1 * np.random.random(PREDICTION_LENGTH)

TIME_SERIES = [
    NAN,
    ZEROES,
    CONSTANT,
    LINEAR,
    EXPONENTIAL,
    RANDOM,
    NEGATIVE_RANDOM,
]


def test_absolute_label():
    for label in TIME_SERIES:
        for forecast in TIME_SERIES:
            data = {"label": label, "mean": forecast}
            actual = absolute_label(data)
            expected = np.abs(label)

            np.testing.assert_almost_equal(actual, expected)


def test_error():
    for label in TIME_SERIES:
        for forecast in TIME_SERIES:
            data = {"label": label, "0.5": forecast}
            actual = error(data, forecast_type="0.5")
            expected = label - forecast

            np.testing.assert_almost_equal(actual, expected)


def test_abs_error():
    for label in TIME_SERIES:
        for forecast in TIME_SERIES:
            data = {"label": label, "0.5": forecast}
            actual = absolute_error(data, forecast_type="0.5")
            expected = np.abs(label - forecast)

            np.testing.assert_almost_equal(actual, expected)


def test_squared_error():
    for label in TIME_SERIES:
        for forecast in TIME_SERIES:
            data = {"label": label, "mean": forecast}
            actual = squared_error(data, forecast_type="mean")
            expected = np.square(label - forecast)

            np.testing.assert_almost_equal(actual, expected)


def test_quantile_loss():
    for label in TIME_SERIES:
        for forecast in TIME_SERIES:
            q = 0.9
            data = {"label": label, str(q): forecast}
            actual = quantile_loss(data, q=q)
            expected = 2 * np.abs(
                (label - forecast) * ((forecast >= label) - q)
            )

            np.testing.assert_almost_equal(actual, expected)


def test_coverage():
    for label in TIME_SERIES:
        for forecast in TIME_SERIES:
            q = 0.9
            data = {"label": label, str(q): forecast}
            actual = coverage(data, q=q)
            expected = label <= forecast

            np.testing.assert_almost_equal(actual, expected)


def test_absolute_percentage_error():
    for label in TIME_SERIES:
        for forecast in TIME_SERIES:
            data = {"label": label, "0.5": forecast}
            actual = absolute_percentage_error(data, forecast_type="0.5")
            expected = np.abs(label - forecast) / np.abs(label)

            np.testing.assert_almost_equal(actual, expected)


def test_symmetric_absolute_percentage_error():
    for label in TIME_SERIES:
        for forecast in TIME_SERIES:
            data = {"label": label, "0.5": forecast}
            actual = symmetric_absolute_percentage_error(
                data, forecast_type="0.5"
            )
            expected = (
                2
                * np.abs(label - forecast)
                / (np.abs(label) + np.abs(forecast))
            )

            np.testing.assert_almost_equal(actual, expected)


def test_scaled_interval_score():
    for label in TIME_SERIES:
        for forecast in TIME_SERIES:
            alpha = 0.05

            # to keep things simple, the following values are rather arbitrary
            lower_quantile = forecast * (alpha / 2)
            upper_quantile = forecast * (1 - alpha / 2)

            # applying `seasonal_error` on the label is not realistic but
            # at least, the seasonal error function gets used this way
            seasonal_err = seasonal_error(label, seasonality=2)

            data = {
                "label": label,
                "mean": forecast,
                "seasonal_error": seasonal_err,
                "0.025": lower_quantile,
                "0.975": upper_quantile,
            }

            actual = scaled_interval_score(data, alpha=alpha)
            expected = (
                upper_quantile
                - lower_quantile
                + 2.0
                / alpha
                * (lower_quantile - label)
                * (label < lower_quantile)
                + 2.0
                / alpha
                * (label - upper_quantile)
                * (label > upper_quantile)
            ) / seasonal_err

            np.testing.assert_almost_equal(actual, expected)


def test_absolute_scaled_error():
    for label in TIME_SERIES:
        for forecast in TIME_SERIES:
            # applying `seasonal_error` on the label is not realistic but
            # at least, the seasonal error function gets used this way
            seasonal_err = seasonal_error(label, seasonality=2)
            data = {
                "label": label,
                "0.5": forecast,
                "seasonal_error": seasonal_err,
            }

            actual = absolute_scaled_error(data, forecast_type="0.5")
            expected = np.abs(label - forecast) / seasonal_err

            np.testing.assert_almost_equal(actual, expected)
