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

from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.trivial.identity import IdentityPredictor


@pytest.mark.parametrize("freq", ["23min", "H", "17H", "9D", "3W", "5M"])
def test_make_evaluation_predictions_zero(freq):
    prediction_length = 5
    ts_length = 50

    predictor = IdentityPredictor(
        prediction_length=prediction_length, num_samples=5
    )
    dataset = [
        dict(
            start=pd.Period("2010-01-01 00:00", freq=freq) + k,
            target=np.arange(start=k, stop=k + ts_length),
        )
        for k in range(10)
    ]

    forecasts, tss = make_evaluation_predictions(dataset, predictor)

    for k, (forecast, ts) in enumerate(zip(forecasts, tss)):
        assert np.allclose(
            forecast.median,
            np.arange(
                start=ts_length - 2 * prediction_length + k,
                stop=ts_length - prediction_length + k,
            ),
        )
        assert isinstance(ts, pd.DataFrame)
        assert isinstance(ts.index, pd.PeriodIndex)
        assert ts.index[0] == dataset[k]["start"]
        assert ts.shape[0] == len(dataset[k]["target"])
        assert np.allclose(
            ts.iloc[:, 0].values, np.arange(start=k, stop=k + ts_length)
        )
