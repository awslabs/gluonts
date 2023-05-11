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

from typing import List

import numpy as np
import pandas as pd
import pytest

from gluonts.evaluation import Evaluator, backtest_metrics
from gluonts.ext.r_forecast import (
    R_IS_INSTALLED,
    RPY2_IS_INSTALLED,
    RForecastPredictor,
)
from gluonts.model.forecast import QuantileForecast

# conditionally skip these tests if `R` and `rpy2` are not installed
if not R_IS_INSTALLED or not RPY2_IS_INSTALLED:
    skip_message = "Skipping test because `R` and `rpy2` are not installed!"
    pytest.skip(skip_message, allow_module_level=True)

freq = "H"
period = 24

## two weeks of data
dataset = [
    {
        "start": pd.Period("1990-01-01 00", freq=freq),
        "target": np.array(
            [
                item
                for i in range(70)
                for item in np.sin(
                    2 * np.pi / period * np.arange(1, period + 1, 1)
                )
            ]
        )
        + np.random.normal(0, 0.5, period * 70)
        + np.array(
            [
                item
                for i in range(10)
                for item in [0 for i in range(5 * 24)]
                + [8 for i in range(4)]
                + [0 for i in range(20)]
                + [8 for i in range(4)]
                + [0 for i in range(20)]
            ]
        ),
    }
]


def no_quantile_crossing(
    forecast: QuantileForecast, quantile_levels: List[float]
):
    sorted_levels = sorted(quantile_levels)
    quantile = forecast.quantile(sorted_levels[0])
    if np.isnan(quantile).any():
        return False

    for level in sorted_levels[1:]:
        prev_quantile = quantile
        quantile = forecast.quantile(level)
        if (quantile < prev_quantile).any():
            return False

    return True


@pytest.mark.parametrize(
    "method",
    [
        "ets",
        "arima",
        "fourier.arima",
    ],
)
def test_model_forecasts(method: str):
    prediction_length = 24 * 7
    quantile_levels = [0.5, 0.9, 0.2, 0.85]

    predictor = RForecastPredictor(
        freq=freq,
        prediction_length=prediction_length,
        period=period,
        method_name=method,
        params={
            "quantiles": quantile_levels,
            "output_types": ["mean", "quantiles"],
        },
    )

    forecast = list(predictor.predict(dataset))[0]

    assert forecast.prediction_length == prediction_length

    for level in quantile_levels:
        assert forecast.quantile(level).shape == (prediction_length,)

    assert no_quantile_crossing(forecast, quantile_levels)


def test_compare_arimas():
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])

    arima = RForecastPredictor(
        freq=freq,
        prediction_length=24 * 7,
        period=period,
        params={
            "quantiles": [0.50, 0.10, 0.90],
            "output_types": ["mean", "quantiles"],
        },
        method_name="arima",
    )

    arima_eval_metrics, _ = backtest_metrics(
        test_dataset=dataset, predictor=arima, evaluator=evaluator
    )

    fourier_arima = RForecastPredictor(
        freq=freq,
        prediction_length=24 * 7,
        period=period,
        params={
            "quantiles": [0.50, 0.10, 0.90],
            "output_types": ["mean", "quantiles"],
        },
        method_name="fourier.arima",
    )
    fourier_arima_eval_metrics, _ = backtest_metrics(
        test_dataset=dataset, predictor=fourier_arima, evaluator=evaluator
    )

    assert fourier_arima_eval_metrics["MASE"] < arima_eval_metrics["MASE"]
