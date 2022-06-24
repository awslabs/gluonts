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
    DayOfMonth,
    DayOfMonthIndex,
    DayOfWeek,
    DayOfWeekIndex,
    DayOfYear,
    DayOfYearIndex,
    HourOfDay,
    HourOfDayIndex,
    MinuteOfHour,
    MinuteOfHourIndex,
    MonthOfYear,
    MonthOfYearIndex,
    SecondOfMinute,
    SecondOfMinuteIndex,
    TimeFeature,
    WeekOfYear,
    WeekOfYearIndex,
)


@pytest.mark.parametrize(
    "feature, index",
    [
        (
            SecondOfMinute(),
            pd.period_range(
                "01-01-2015 00:00:00", periods=60 * 2 * 24, freq="5S"
            ),
        ),
        (
            MinuteOfHour(),
            pd.period_range(
                "01-01-2015 00:00:00", periods=60 * 2 * 24, freq="1min"
            ),
        ),
        (
            HourOfDay(),
            pd.period_range("01-01-2015 00:00:00", periods=14 * 24, freq="1h"),
        ),
        (
            DayOfWeek(),
            pd.period_range("01-01-2015", periods=365 * 5, freq="D"),
        ),
        (
            DayOfMonth(),
            pd.period_range("01-01-2015", periods=365 * 5, freq="D"),
        ),
        (
            DayOfYear(),
            pd.period_range("01-01-2015", periods=365 * 5, freq="D"),
        ),
        (
            WeekOfYear(),
            pd.period_range("01-01-2015", periods=53 * 5, freq="W"),
        ),
        (
            MonthOfYear(),
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
            SecondOfMinuteIndex(),
            pd.period_range(
                "01-01-2015 00:00:00", periods=60 * 2 * 24, freq="1S"
            ),
            60,
        ),
        (
            MinuteOfHourIndex(),
            pd.period_range(
                "01-01-2015 00:00:00", periods=60 * 2 * 24, freq="1min"
            ),
            60,
        ),
        (
            HourOfDayIndex(),
            pd.period_range("01-01-2015 00:00:00", periods=14 * 24, freq="1h"),
            24,
        ),
        (
            DayOfWeekIndex(),
            pd.period_range("01-01-2015", periods=365 * 5, freq="D"),
            7,
        ),
        (
            DayOfMonthIndex(),
            pd.period_range("01-01-2015", periods=365 * 5, freq="D"),
            31,
        ),
        (
            DayOfYearIndex(),
            pd.period_range("01-01-2015", periods=365 * 5, freq="D"),
            366,
        ),
        (
            WeekOfYearIndex(),
            pd.period_range("01-01-2015", periods=53 * 5, freq="W"),
            53,
        ),
        (
            MonthOfYearIndex(),
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
