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

import pandas as pd
import pytest
import numpy as np

from gluonts.dataset import Dataset
from gluonts.model import QuantileForecast
from gluonts.model.statsforecast import (
    ModelConfig,
    StatsForecastPredictor,
    ADIDAPredictor,
    AutoARIMAPredictor,
    AutoCESPredictor,
    CrostonClassicPredictor,
    CrostonOptimizedPredictor,
    CrostonSBAPredictor,
    ETSPredictor,
    IMAPAPredictor,
    TSBPredictor,
)


@pytest.mark.parametrize(
    "quantile_levels, intervals, forecast_keys, statsforecast_keys",
    [
        (
            [0.5, 0.2, 0.6, 0.9, 0.8, 0.1],
            [0, 20, 60, 80],
            ["mean", "0.5", "0.2", "0.6", "0.9", "0.8", "0.1"],
            ["mean", "lo-0", "lo-60", "hi-20", "hi-80", "hi-60", "lo-80"],
        ),
    ],
)
def test_model_config(
    quantile_levels, intervals, forecast_keys, statsforecast_keys
):
    config = ModelConfig(quantile_levels=quantile_levels)
    assert config.intervals == intervals
    assert config.forecast_keys == forecast_keys
    assert config.statsforecast_keys == statsforecast_keys


@pytest.mark.parametrize(
    "predictor",
    [
        ADIDAPredictor(prediction_length=3),
        AutoARIMAPredictor(
            prediction_length=3,
            quantile_levels=[0.5, 0.1, 0.9, 0.95],
        ),
        AutoCESPredictor(prediction_length=3, season_length=12),
        CrostonClassicPredictor(prediction_length=3),
        CrostonOptimizedPredictor(prediction_length=3),
        CrostonSBAPredictor(prediction_length=3),
        ETSPredictor(prediction_length=3, season_length=12),
        IMAPAPredictor(prediction_length=3),
        TSBPredictor(prediction_length=3, alpha_d=0.5, alpha_p=0.5),
    ],
)
@pytest.mark.parametrize(
    "dataset",
    [
        [
            dict(
                start=pd.Period("2021-02-03 00", freq="H"),
                target=np.random.normal(loc=10, scale=0.5, size=(100,)),
            )
        ]
    ],
)
def test_predictor_working(
    predictor: StatsForecastPredictor, dataset: Dataset
):
    for forecast in predictor.predict(dataset):
        assert isinstance(forecast, QuantileForecast)
        assert len(forecast.mean) == predictor.prediction_length
        assert predictor.config.forecast_keys == forecast.forecast_keys
