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

from gluonts.model.seasonal_agg import SeasonalAggregatePredictor
from gluonts.transform.feature import LastValueImputation, LeavesMissingValues

FREQ = "D"
START_DATE = "2023"


def get_prediction(
    target,
    prediction_length=1,
    season_length=1,
    num_seasons=1,
    agg_fun=np.nanmean,
    imputation_method=LastValueImputation(),
):
    pred = SeasonalAggregatePredictor(
        prediction_length=prediction_length,
        season_length=season_length,
        num_seasons=num_seasons,
        agg_fun=agg_fun,
        imputation_method=imputation_method,
    )
    item = {
        "target": np.asarray(target),
        "start": pd.Period(START_DATE, freq=FREQ),
    }
    forecast = pred.predict_item(item)

    return forecast


@pytest.mark.parametrize(
    "data, expected_output, prediction_length, season_length, "
    "num_seasons, agg_fun, imputation_method",
    [
        # same as seasonal naive
        ([1, 1, 1], [1], 1, 1, 1, np.nanmean, LastValueImputation()),
        (
            [1, 10, 2, 20],
            [1.5, 15],
            2,
            2,
            2,
            np.nanmean,
            LastValueImputation(),
        ),
        # check predictions repeat seasonally
        (
            [1, 10, 2, 20],
            [1.5, 15, 1.5, 15],
            4,
            2,
            2,
            np.nanmean,
            LastValueImputation(),
        ),
        (
            [1, 10, 2, 20],
            [1.5, 15, 1.5],
            3,
            2,
            2,
            np.nanmean,
            LastValueImputation(),
        ),
        # check `nanmedian`
        (
            [1, 10, 2, 20, 3, 30],
            [2, 20, 2, 20],
            4,
            2,
            3,
            np.nanmedian,
            LastValueImputation(),
        ),
        (
            [1, 10, 2, 20, 3, 30],
            [2, 20, 2],
            3,
            2,
            3,
            np.nanmedian,
            LastValueImputation(),
        ),
        # check `nanmax`
        (
            [1, 10, 2, 20, 3, 30],
            [3, 30, 3, 30],
            4,
            2,
            3,
            np.nanmax,
            LastValueImputation(),
        ),
        # check `nanmin`
        (
            [1, 10, 2, 20, 3, 30],
            [1, 10, 1, 10],
            4,
            2,
            3,
            np.nanmin,
            LastValueImputation(),
        ),
        # data is shorter than season length
        ([1, 2, 3], [2], 1, 4, 1, np.nanmean, LastValueImputation()),
        ([10, 1, 100], [10], 1, 4, 1, np.nanmedian, LastValueImputation()),
        ([10, 1, 100], [100], 1, 4, 1, np.nanmax, LastValueImputation()),
        ([10, 1, 100], [1], 1, 4, 1, np.nanmin, LastValueImputation()),
        # data not available for all seasons
        ([1, 2, 3, 4, 5], [3] * 4, 4, 4, 2, np.nanmean, LastValueImputation()),
        (
            [10, 20, 40, 50, 21],
            [21] * 4,
            4,
            4,
            2,
            np.nanmedian,
            LastValueImputation(),
        ),
        (
            [10, 20, 40, 50, 21],
            [50] * 4,
            4,
            4,
            2,
            np.nanmax,
            LastValueImputation(),
        ),
        (
            [10, 20, 40, 50, 21],
            [10] * 4,
            4,
            4,
            2,
            np.nanmin,
            LastValueImputation(),
        ),
        # missing values with imputation
        ([np.nan], [0], 1, 1, 2, np.nanmean, LastValueImputation()),
        ([np.nan], [0], 1, 1, 2, np.nanmedian, LastValueImputation()),
        ([1, 4, np.nan], [3], 1, 3, 2, np.nanmean, LastValueImputation()),
        ([1, 4, np.nan], [4], 1, 3, 2, np.nanmedian, LastValueImputation()),
        (
            [1, 10, np.nan, 1, 10, np.nan],
            [1, 10, 10],
            3,
            3,
            2,
            np.nanmean,
            LastValueImputation(),
        ),
        (
            [1, 10, np.nan, 1, 10, np.nan],
            [1, 10, 10],
            3,
            3,
            2,
            np.nanmedian,
            LastValueImputation(),
        ),
        (
            [1, 10, np.nan, 1, 10, np.nan],
            [1, 10, 10, 1, 10],
            5,
            3,
            2,
            np.nanmax,
            LastValueImputation(),
        ),
        (
            [1, 10, np.nan, 1, 10, np.nan],
            [1, 10, 10, 1, 10],
            5,
            3,
            2,
            np.nanmin,
            LastValueImputation(),
        ),
        # missing values without imputation
        ([1, 3, np.nan], [np.nan], 1, 1, 1, np.nanmean, LeavesMissingValues()),
        (
            [1, 3, np.nan],
            [np.nan],
            1,
            1,
            1,
            np.nanmedian,
            LeavesMissingValues(),
        ),
        ([1, 3, np.nan], [np.nan], 1, 1, 1, np.nanmax, LeavesMissingValues()),
        ([1, 3, np.nan], [np.nan], 1, 1, 1, np.nanmin, LeavesMissingValues()),
        (
            [1, 3, np.nan],
            [np.nan] * 2,
            2,
            1,
            1,
            np.nanmean,
            LeavesMissingValues(),
        ),
        (
            [1, 3, np.nan],
            [np.nan] * 2,
            2,
            1,
            1,
            np.nanmedian,
            LeavesMissingValues(),
        ),
        (
            [1, 3, np.nan],
            [np.nan] * 2,
            2,
            1,
            1,
            np.nanmax,
            LeavesMissingValues(),
        ),
        (
            [1, 3, np.nan],
            [np.nan] * 2,
            2,
            1,
            1,
            np.nanmin,
            LeavesMissingValues(),
        ),
        ([1, 3, np.nan], [3], 1, 1, 2, np.nanmean, LeavesMissingValues()),
        ([1, 3, np.nan], [3], 1, 1, 2, np.nanmedian, LeavesMissingValues()),
        ([1, 3, np.nan], [3], 1, 1, 2, np.nanmax, LeavesMissingValues()),
        ([1, 3, np.nan], [3], 1, 1, 2, np.nanmin, LeavesMissingValues()),
        ([1, 3, np.nan], [3], 1, 2, 1, np.nanmean, LeavesMissingValues()),
        ([1, 3, np.nan], [3], 1, 2, 1, np.nanmedian, LeavesMissingValues()),
        ([1, 3, np.nan], [3], 1, 2, 1, np.nanmax, LeavesMissingValues()),
        ([1, 3, np.nan], [3], 1, 2, 1, np.nanmin, LeavesMissingValues()),
        (
            [1, 3, np.nan],
            [3, np.nan],
            2,
            2,
            1,
            np.nanmean,
            LeavesMissingValues(),
        ),
        (
            [1, 3, np.nan],
            [3, np.nan],
            2,
            2,
            1,
            np.nanmedian,
            LeavesMissingValues(),
        ),
        (
            [1, 3, np.nan],
            [3, np.nan],
            2,
            2,
            1,
            np.nanmax,
            LeavesMissingValues(),
        ),
        (
            [1, 3, np.nan],
            [3, np.nan],
            2,
            2,
            1,
            np.nanmin,
            LeavesMissingValues(),
        ),
        # check if `nanmean` works when some seasons have missing values
        ([1, 3, np.nan], [3, 3], 2, 1, 2, np.nanmean, LeavesMissingValues()),
        (
            [1, 3, np.nan],
            [3, 3, 3],
            3,
            1,
            2,
            np.nanmean,
            LeavesMissingValues(),
        ),
        # check if `mean` works when some seasons have missing values
        (
            [1, 3, np.nan],
            [np.nan] * 2,
            2,
            1,
            2,
            np.mean,
            LeavesMissingValues(),
        ),
        (
            [1, 3, np.nan],
            [np.nan] * 3,
            3,
            1,
            2,
            np.mean,
            LeavesMissingValues(),
        ),
        # check if `nanmedian` works when some seasons have missing values
        ([1, 3, np.nan], [3, 3], 2, 1, 2, np.nanmedian, LeavesMissingValues()),
        (
            [1, 3, np.nan],
            [3, 3, 3],
            3,
            1,
            2,
            np.nanmedian,
            LeavesMissingValues(),
        ),
        # check if `nanmax` works when some seasons have missing values
        ([1, 3, np.nan], [3, 3], 2, 1, 2, np.nanmax, LeavesMissingValues()),
        ([1, 3, np.nan], [3, 3, 3], 3, 1, 2, np.nanmax, LeavesMissingValues()),
        # check if `nanmin` works when some seasons have missing values
        ([1, 3, np.nan], [3, 3], 2, 1, 2, np.nanmin, LeavesMissingValues()),
        ([1, 3, np.nan], [3, 3, 3], 3, 1, 2, np.nanmin, LeavesMissingValues()),
        # check if `mean` works when some seasons have missing values
        (
            [1, 3, np.nan],
            [np.nan] * 2,
            2,
            1,
            2,
            np.median,
            LeavesMissingValues(),
        ),
        (
            [1, 3, np.nan],
            [np.nan] * 3,
            3,
            1,
            2,
            np.median,
            LeavesMissingValues(),
        ),
        # check if `median` works when some seasons have missing values
        (
            [1, 3, np.nan],
            [np.nan] * 2,
            2,
            1,
            2,
            np.median,
            LeavesMissingValues(),
        ),
        (
            [1, 3, np.nan],
            [np.nan] * 3,
            3,
            1,
            2,
            np.median,
            LeavesMissingValues(),
        ),
        # check if `max` works when some seasons have missing values
        ([1, 3, np.nan], [np.nan] * 2, 2, 1, 2, np.max, LeavesMissingValues()),
        ([1, 3, np.nan], [np.nan] * 3, 3, 1, 2, np.max, LeavesMissingValues()),
        # check if `min` works when some seasons have missing values
        ([1, 3, np.nan], [np.nan] * 2, 2, 1, 2, np.min, LeavesMissingValues()),
        ([1, 3, np.nan], [np.nan] * 3, 3, 1, 2, np.min, LeavesMissingValues()),
    ],
)
def test_predictor(
    data,
    expected_output,
    prediction_length,
    season_length,
    num_seasons,
    agg_fun,
    imputation_method,
):
    prediction = get_prediction(
        data,
        prediction_length=prediction_length,
        season_length=season_length,
        num_seasons=num_seasons,
        agg_fun=agg_fun,
        imputation_method=imputation_method,
    )
    assert prediction.samples.shape == (1, prediction_length)

    np.testing.assert_equal(prediction.mean, expected_output)
