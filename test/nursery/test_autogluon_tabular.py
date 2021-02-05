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

from typing import List, Optional

import pytest
import numpy as np
import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas
from gluonts.nursery.autogluon_tabular import get_features_dataframe
from gluonts.nursery.autogluon_tabular import (
    TabularEstimator,
    LocalTabularPredictor,
)


@pytest.mark.parametrize(
    "series, lags, past_data, expected_df",
    [
        (
            pd.Series(
                list(range(5)),
                index=pd.date_range(
                    "2020-12-31 22:00:00", freq="H", periods=5
                ),
            ),
            [1, 2, 5],
            None,
            pd.DataFrame(
                {
                    "year": [2020, 2020, 2021, 2021, 2021],
                    "month_of_year": [12, 12, 1, 1, 1],
                    "day_of_week": [3, 3, 4, 4, 4],
                    "hour_of_day": [22, 23, 0, 1, 2],
                    "holiday_indicator": [False, False, True, False, False],
                    "lag_1": [np.nan, 0, 1, 2, 3],
                    "lag_2": [np.nan, np.nan, 0, 1, 2],
                    "lag_5": [np.nan, np.nan, np.nan, np.nan, np.nan],
                    "target": list(range(5)),
                },
                index=pd.date_range(
                    "2020-12-31 22:00:00", freq="H", periods=5
                ),
            ),
        ),
        (
            pd.Series(
                list(range(5)),
                index=pd.date_range(
                    "2020-12-31 22:00:00", freq="H", periods=5
                ),
            ),
            [1, 2, 5],
            pd.Series(
                list(range(5)),
                index=pd.date_range(
                    "2020-12-31 16:00:00", freq="H", periods=5
                ),
            ),
            pd.DataFrame(
                {
                    "year": [2020, 2020, 2021, 2021, 2021],
                    "month_of_year": [12, 12, 1, 1, 1],
                    "day_of_week": [3, 3, 4, 4, 4],
                    "hour_of_day": [22, 23, 0, 1, 2],
                    "holiday_indicator": [False, False, True, False, False],
                    "lag_1": [np.nan, 0, 1, 2, 3],
                    "lag_2": [4, np.nan, 0, 1, 2],
                    "lag_5": [1, 2, 3, 4, np.nan],
                    "target": list(range(5)),
                },
                index=pd.date_range(
                    "2020-12-31 22:00:00", freq="H", periods=5
                ),
            ),
        ),
    ],
)
def test_get_features_dataframe(
    series: pd.Series,
    lags: List[int],
    past_data: Optional[pd.Series],
    expected_df: pd.DataFrame,
):
    assert expected_df.equals(
        get_features_dataframe(series, lags=lags, past_data=past_data)
    )


@pytest.mark.parametrize(
    "dataset, freq, prediction_length",
    [
        (
            ListDataset(
                [
                    {
                        "start": pd.Timestamp(
                            "1750-01-04 00:00:00", freq="W-SUN"
                        ),
                        "target": np.array(
                            [
                                1089.2,
                                1078.91,
                                1099.88,
                                35790.55,
                                34096.95,
                                34906.95,
                            ],
                        ),
                    },
                    {
                        "start": pd.Timestamp(
                            "1750-01-04 00:00:00", freq="W-SUN"
                        ),
                        "target": np.array(
                            [
                                1099.2,
                                1098.91,
                                1069.88,
                                35990.55,
                                34076.95,
                                34766.95,
                            ],
                        ),
                    },
                ],
                freq="W-SUN",
            ),
            "W-SUN",
            2,
        )
    ],
)
@pytest.mark.parametrize("lags", [[], [1, 2, 5]])
def test_local_tabular_predictor(
    dataset, freq, prediction_length: int, lags: List[int]
):
    predictor = LocalTabularPredictor(
        freq=freq,
        prediction_length=prediction_length,
        lags=lags,
        time_limits=10,
    )
    forecasts_it = predictor.predict(dataset)
    forecasts = list(forecasts_it)

    for entry, forecast in zip(dataset, forecasts):
        ts = to_pandas(entry)
        start_timestamp = ts.index[-1] + pd.tseries.frequencies.to_offset(freq)
        assert forecast.samples.shape[1] == prediction_length
        assert forecast.start_date == start_timestamp


@pytest.mark.parametrize(
    "dataset, freq, prediction_length",
    [
        (
            ListDataset(
                [
                    {
                        "start": pd.Timestamp(
                            "1750-01-04 00:00:00", freq="W-SUN"
                        ),
                        "target": np.array(
                            [
                                1089.2,
                                1078.91,
                                1099.88,
                                35790.55,
                                34096.95,
                                34906.95,
                            ],
                        ),
                    },
                    {
                        "start": pd.Timestamp(
                            "1750-01-04 00:00:00", freq="W-SUN"
                        ),
                        "target": np.array(
                            [
                                1099.2,
                                1098.91,
                                1069.88,
                                35990.55,
                                34076.95,
                                34766.95,
                            ],
                        ),
                    },
                ],
                freq="W-SUN",
            ),
            "W-SUN",
            2,
        )
    ],
)
@pytest.mark.parametrize("lags", [[], [1, 2, 5]])
def test_tabular_estimator(
    dataset, freq, prediction_length: int, lags: List[int]
):
    estimator = TabularEstimator(
        freq=freq,
        prediction_length=prediction_length,
        lags=lags,
        time_limits=10,
    )

    predictor = estimator.train(dataset)

    forecasts_serial = list(predictor._predict_serial(dataset))
    forecasts_batch = list(predictor._predict_batch(dataset, batch_size=2))

    def check_consistency(entry, f1, f2):
        ts = to_pandas(entry)
        start_timestamp = ts.index[-1] + pd.tseries.frequencies.to_offset(freq)
        assert f1.samples.shape == (1, prediction_length)
        assert f1.start_date == start_timestamp
        assert f2.samples.shape == (1, prediction_length)
        assert f2.start_date == start_timestamp
        assert np.allclose(f1.samples, f2.samples)

    for entry, f1, f2 in zip(dataset, forecasts_serial, forecasts_batch):
        check_consistency(entry, f1, f2)

    if not predictor.auto_regression:
        forecasts_batch_autoreg = list(
            predictor._predict_batch_autoreg(dataset)
        )
        for entry, f1, f2 in zip(
            dataset, forecasts_serial, forecasts_batch_autoreg
        ):
            check_consistency(entry, f1, f2)
