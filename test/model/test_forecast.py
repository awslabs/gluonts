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
import pandas as pd
import pytest

from gluonts.model.forecast import (
    QuantileForecast,
    SampleForecast,
    LinearInterpolation,
    ExponentialTailApproximation,
)

QUANTILES = np.arange(1, 100) / 100
SAMPLES = np.arange(101).reshape(101, 1) / 100
FREQ = "1D"
START_DATE = pd.Period("2017 01-01 12:00", FREQ)

FORECASTS = {
    "QuantileForecast": QuantileForecast(
        forecast_arrays=QUANTILES.reshape(-1, 1),
        start_date=START_DATE,
        forecast_keys=np.array(QUANTILES, str),
    ),
    "SampleForecast": SampleForecast(
        samples=SAMPLES,
        start_date=START_DATE,
    ),
}


@pytest.mark.parametrize("name", FORECASTS.keys())
def test_forecast(name):
    forecast = FORECASTS[name]

    def percentile(value):
        return f"p{int(round(value * 100)):02d}"

    num_samples, pred_length = SAMPLES.shape

    for quantile in QUANTILES:
        test_cases = [quantile, str(quantile), percentile(quantile)]
        for quant_pred in map(forecast.quantile, test_cases):
            assert np.isclose(
                quant_pred[0], quantile
            ), f"Expected {percentile(quantile)} quantile {quantile}. Obtained {quant_pred}."

    assert forecast.prediction_length == 1
    assert len(forecast.index) == pred_length
    assert forecast.index[0] == START_DATE

    forecast.plot()


def test_mean_only_forecast():
    forecast = QuantileForecast(
        forecast_arrays=np.ones(shape=(1, 12)),
        start_date=pd.Period("2022-03-04 00", freq="H"),
        forecast_keys=["mean"],
    )

    assert forecast.prediction_length == 12
    assert len(forecast.index) == 12
    assert forecast.index[0] == pd.Period("2022-03-04 00", freq="H")

    for level in [0.1, 0.5, 0.7]:
        assert np.isnan(forecast.quantile(level)).all()

    assert np.equal(forecast.mean, 1).all()


@pytest.mark.parametrize(
    "forecast, exp_index",
    [
        (
            SampleForecast(
                samples=np.random.normal(size=(100, 7, 3)),
                start_date=pd.Period("2020-01-01 00:00:00", freq="2D"),
            ),
            pd.period_range(
                start=pd.Period("2020-01-01 00:00:00", freq="2D"),
                periods=7,
                freq="2D",
            ),
        ),
    ],
)
def test_forecast_multivariate(forecast, exp_index):
    assert forecast.prediction_length == len(exp_index)
    assert np.all(forecast.index == exp_index)


def test_linear_interpolation() -> None:
    tol = 1e-7
    x_coord = [0.1, 0.5, 0.9]
    y_coord = [
        np.array([0.1, 0.5, 1]),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.25, 0.5, 0.9]),
    ]
    linear_interpolation = LinearInterpolation(x_coord, y_coord)
    x = 0.75
    exact = y_coord[1] + (x - x_coord[1]) * (y_coord[2] - y_coord[1]) / (
        x_coord[2] - x_coord[1]
    )
    assert np.all(np.abs(exact - linear_interpolation(x)) <= tol)


def test_exponential_left_tail_approximation() -> None:
    tol = 1e-5
    x_coord = [0.1, 0.5, 0.9]
    y_coord = [
        np.array([0.1, 0.5, 1]),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.25, 0.5, 0.9]),
    ]
    x = 0.01
    beta_inv = np.array([0.55920144, 0.9320024, 1.24266987])
    exact = beta_inv * np.log(x / x_coord[1]) + y_coord[1]
    exp_tail_approximation = ExponentialTailApproximation(x_coord, y_coord)
    assert np.all(np.abs(exact - exp_tail_approximation.left(x)) <= tol)


def test_exponential_right_tail_approximation() -> None:
    tol = 1e-5
    x_coord = [0.1, 0.5, 0.9]
    y_coord = [
        np.array([0.1, 0.5, 1]),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.25, 0.5, 0.9]),
    ]
    x = 0.99
    beta_inv = np.array([-0.4660012, -0.9320024, -1.30480336])
    exact = beta_inv * np.log((1 - x_coord[1]) / (1 - x)) + y_coord[1]
    exp_tail_approximation = ExponentialTailApproximation(x_coord, y_coord)
    assert np.all(np.abs(exact - exp_tail_approximation.right(x)) <= tol)
