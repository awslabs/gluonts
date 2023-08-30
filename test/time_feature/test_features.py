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

from gluonts.time_feature import (
    Constant,
    TimeFeature,
    day_of_month,
    day_of_month_index,
    day_of_week,
    day_of_week_index,
    day_of_year,
    day_of_year_index,
    hour_of_day,
    hour_of_day_index,
    minute_of_hour,
    minute_of_hour_index,
    month_of_year,
    month_of_year_index,
    second_of_minute,
    second_of_minute_index,
    week_of_year,
    week_of_year_index,
)


@pytest.mark.parametrize(
    "feature, index",
    [
        (
            second_of_minute,
            pd.period_range(
                "01-01-2015 00:00:00", periods=60 * 2 * 24, freq="5S"
            ),
        ),
        (
            minute_of_hour,
            pd.period_range(
                "01-01-2015 00:00:00", periods=60 * 2 * 24, freq="1min"
            ),
        ),
        (
            hour_of_day,
            pd.period_range("01-01-2015 00:00:00", periods=14 * 24, freq="1h"),
        ),
        (
            day_of_week,
            pd.period_range("01-01-2015", periods=365 * 5, freq="D"),
        ),
        (
            day_of_month,
            pd.period_range("01-01-2015", periods=365 * 5, freq="D"),
        ),
        (
            day_of_year,
            pd.period_range("01-01-2015", periods=365 * 5, freq="D"),
        ),
        (
            week_of_year,
            pd.period_range("01-01-2015", periods=53 * 5, freq="W"),
        ),
        (
            month_of_year,
            pd.period_range("01-01-2015", periods=12 * 5, freq="M"),
        ),
        (Constant(), pd.period_range("01-01-2015", periods=5, freq="A")),
    ],
)
def test_feature_normalized_bounds(
    feature: TimeFeature, index: pd.PeriodIndex
):
    values = feature(index)
    assert isinstance(values, np.ndarray)
    for v in values:
        assert -0.5 <= v <= 0.5


@pytest.mark.parametrize(
    "feature, index, cardinality",
    [
        (
            second_of_minute_index,
            pd.period_range(
                "01-01-2015 00:00:00", periods=60 * 2 * 24, freq="1S"
            ),
            60,
        ),
        (
            minute_of_hour_index,
            pd.period_range(
                "01-01-2015 00:00:00", periods=60 * 2 * 24, freq="1min"
            ),
            60,
        ),
        (
            hour_of_day_index,
            pd.period_range("01-01-2015 00:00:00", periods=14 * 24, freq="1h"),
            24,
        ),
        (
            day_of_week_index,
            pd.period_range("01-01-2015", periods=365 * 5, freq="D"),
            7,
        ),
        (
            day_of_month_index,
            pd.period_range("01-01-2015", periods=365 * 5, freq="D"),
            31,
        ),
        (
            day_of_year_index,
            pd.period_range("01-01-2015", periods=365 * 5, freq="D"),
            366,
        ),
        (
            week_of_year_index,
            pd.period_range("01-01-2015", periods=53 * 5, freq="W"),
            53,
        ),
        (
            month_of_year_index,
            pd.period_range("01-01-2015", periods=12 * 5, freq="M"),
            12,
        ),
    ],
)
def test_feature_unnormalized_bounds(
    feature: TimeFeature, index: pd.DatetimeIndex, cardinality: int
):
    values = feature(index)
    assert isinstance(values, np.ndarray)
    counts = [0] * cardinality
    for v in values:
        assert 0 <= int(v) < cardinality
        counts[int(v)] += 1
    assert all(c > 0 for c in counts)
