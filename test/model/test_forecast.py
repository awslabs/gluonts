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
from gluonts.model.forecast import QuantileForecast, SampleForecast

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
            assert (
                quant_pred[0] == quantile
            ), f"Expected {quantile} quantile {quantile}. Obtained {quant_pred}."

    assert forecast.prediction_length == 1
    assert len(forecast.index) == pred_length
    assert forecast.index[0] == pd.Timestamp(START_DATE)
