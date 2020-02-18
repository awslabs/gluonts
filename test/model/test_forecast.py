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
import mxnet as mx

# First-party imports
from gluonts.model.forecast import (
    QuantileForecast,
    SampleForecast,
    DistributionForecast,
)

from gluonts.distribution import Uniform

QUANTILES = np.arange(1, 100) / 100
SAMPLES = np.arange(101).reshape(101, 1) / 100
START_DATE = pd.Timestamp(2017, 1, 1, 12)
FREQ = "1D"

FORECASTS = {
    "QuantileForecast": QuantileForecast(
        forecast_arrays=QUANTILES.reshape(-1, 1),
        start_date=START_DATE,
        forecast_keys=np.array(QUANTILES, str),
        freq=FREQ,
    ),
    "SampleForecast": SampleForecast(
        samples=SAMPLES, start_date=START_DATE, freq=FREQ
    ),
    "DistributionForecast": DistributionForecast(
        distribution=Uniform(low=mx.nd.zeros(1), high=mx.nd.ones(1)),
        start_date=START_DATE,
        freq=FREQ,
    ),
}


@pytest.mark.parametrize("name", FORECASTS.keys())
def test_Forecast(name):
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
    assert forecast.index[0] == pd.Timestamp(START_DATE)


def test_DistributionForecast():
    forecast = DistributionForecast(
        distribution=Uniform(
            low=mx.nd.array([0.0, 0.0]), high=mx.nd.array([1.0, 2.0])
        ),
        start_date=START_DATE,
        freq=FREQ,
    )

    def percentile(value):
        return f"p{int(round(value * 100)):02d}"

    for quantile in QUANTILES:
        test_cases = [quantile, str(quantile), percentile(quantile)]
        for quant_pred in map(forecast.quantile, test_cases):
            expected = quantile * np.array([1.0, 2.0])
            assert np.allclose(
                quant_pred, expected
            ), f"Expected {percentile(quantile)} quantile {quantile}. Obtained {quant_pred}."

    pred_length = 2
    assert forecast.prediction_length == pred_length
    assert len(forecast.index) == pred_length
    assert forecast.index[0] == pd.Timestamp(START_DATE)


@pytest.mark.parametrize(
    "forecast, exp_index",
    [
        (
            SampleForecast(
                samples=np.random.normal(size=(100, 7, 3)),
                start_date=pd.Timestamp("2020-01-01 00:00:00"),
                freq="1D",
            ),
            pd.date_range(
                start=pd.Timestamp("2020-01-01 00:00:00"),
                freq="1D",
                periods=7,
            ),
        ),
        (
            DistributionForecast(
                Uniform(
                    low=mx.nd.zeros(shape=(5, 2)),
                    high=mx.nd.ones(shape=(5, 2)),
                ),
                start_date=pd.Timestamp("2020-01-01 00:00:00"),
                freq="W",
            ),
            pd.date_range(
                start=pd.Timestamp("2020-01-01 00:00:00"), freq="W", periods=5,
            ),
        ),
    ],
)
def test_forecast_multivariate(forecast, exp_index):
    assert forecast.prediction_length == len(exp_index)
    assert np.all(forecast.index == exp_index)
