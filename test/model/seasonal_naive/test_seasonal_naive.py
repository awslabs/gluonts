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

from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.transform.feature import LastValueImputation, LeavesMissingValues

FREQ = "D"
START_DATE = "2023"


def get_prediction(
    target,
    prediction_length=1,
    season_length=1,
    imputation_method=LastValueImputation(),
):
    pred = SeasonalNaivePredictor(
        freq=FREQ,
        prediction_length=prediction_length,
        season_length=season_length,
        imputation_method=imputation_method,
    )
    item = {
        "target": np.asarray(target),
        "start": pd.Period(START_DATE, freq=FREQ),
    }
    forecast = pred.predict_item(item)

    return forecast


@pytest.mark.parametrize(
    "data, expected_output, prediction_length, season_length, imputation_method",
    [
        ([1, 1, 1], [1], 1, 1, LastValueImputation()),
        ([1, 1, 1], [1, 1], 2, 1, LastValueImputation()),
        ([1, 1, 1], [1, 1, 1], 3, 1, LastValueImputation()),
        ([1, 1, 1], [1], 1, 2, LastValueImputation()),
        ([1, 1, 1], [1, 1], 2, 2, LastValueImputation()),
        ([1, 1, 1], [1, 1, 1], 3, 2, LastValueImputation()),
        ([1, 1, 1], [1], 1, 3, LastValueImputation()),
        ([1, 1, 1], [1, 1], 2, 3, LastValueImputation()),
        ([1, 1, 1], [1, 1, 1], 3, 3, LastValueImputation()),
        ([np.nan], [0], 1, 1, LastValueImputation()),
        ([1, 3, np.nan], [1], 1, 3, LastValueImputation()),
        ([1, 3, np.nan], [1, 3], 2, 3, LastValueImputation()),
        ([1, 3, np.nan], [1, 3, 3], 3, 3, LastValueImputation()),
        ([1, 2, 3], [3], 1, 1, LastValueImputation()),
        ([1, 2, 3], [3, 3], 2, 1, LastValueImputation()),
        ([1, 2, 3], [3, 3, 3], 3, 1, LastValueImputation()),
        ([1, 2, 3], [2], 1, 2, LastValueImputation()),
        ([1, 2, 3], [2, 3], 2, 2, LastValueImputation()),
        ([1, 2, 3], [2, 3, 2], 3, 2, LastValueImputation()),
        ([1, 2, 3], [1], 1, 3, LastValueImputation()),
        ([1, 2, 3], [1, 2], 2, 3, LastValueImputation()),
        ([1, 2, 3], [1, 2, 3], 3, 3, LastValueImputation()),
        ([1, 1, 1], [1], 1, None, LastValueImputation()),
        ([1, 1, 1], [1, 1], 2, None, LastValueImputation()),
        ([1, 1, 1], [1, 1, 1], 3, None, LastValueImputation()),
        ([1, 3, np.nan], [3], 1, None, LastValueImputation()),
        ([1, 3, np.nan], [3, 3], 2, None, LastValueImputation()),
        ([1, 3, np.nan], [3, 3, 3], 3, None, LastValueImputation()),
        ([1, 3, np.nan], [np.nan], 1, 1, LeavesMissingValues()),
        ([1, 3, np.nan], [np.nan] * 2, 2, 1, LeavesMissingValues()),
        ([1, 3, np.nan], [np.nan] * 3, 3, 1, LeavesMissingValues()),
        ([1, 3, np.nan], [3], 1, 2, LeavesMissingValues()),
        ([1, 3, np.nan], [3, np.nan], 2, 2, LeavesMissingValues()),
        ([1, 3, np.nan], [3, np.nan, 3], 3, 2, LeavesMissingValues()),
    ],
)
def test_predictor(
    data, expected_output, prediction_length, season_length, imputation_method
):
    prediction = get_prediction(
        data,
        prediction_length=prediction_length,
        season_length=season_length,
        imputation_method=imputation_method,
    )
    assert prediction.samples.shape == (1, prediction_length)

    np.testing.assert_equal(prediction.mean, expected_output)
