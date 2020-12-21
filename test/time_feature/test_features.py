import numpy as np
import pandas as pd
import pytest

from gluonts.time_feature import (
    TimeFeature,
    MinuteOfHour,
    HourOfDay,
    DayOfWeek,
    DayOfMonth,
    DayOfYear,
    WeekOfYear,
    MonthOfYear,
)


@pytest.mark.parametrize(
    "feature, index",
    [
        (
            MinuteOfHour(),
            pd.date_range(
                "01-01-2015 00:00:00", periods=60 * 2 * 24, freq="1min"
            ),
        ),
        (
            HourOfDay(),
            pd.date_range("01-01-2015 00:00:00", periods=14 * 24, freq="1h"),
        ),
        (DayOfWeek(), pd.date_range("01-01-2015", periods=365 * 5, freq="D")),
        (DayOfMonth(), pd.date_range("01-01-2015", periods=365 * 5, freq="D")),
        (DayOfYear(), pd.date_range("01-01-2015", periods=365 * 5, freq="D")),
        (WeekOfYear(), pd.date_range("01-01-2015", periods=53 * 5, freq="W")),
        (MonthOfYear(), pd.date_range("01-01-2015", periods=12 * 5, freq="M")),
    ],
)
def test_feature_normalized_bounds(
    feature: TimeFeature, index: pd.DatetimeIndex
):
    values = feature(index)
    for v in values:
        assert -0.5 <= v <= 0.5


@pytest.mark.parametrize(
    "feature, index, cardinality",
    [
        (
            MinuteOfHour(normalized=False),
            pd.date_range(
                "01-01-2015 00:00:00", periods=60 * 2 * 24, freq="1min"
            ),
            60,
        ),
        (
            HourOfDay(normalized=False),
            pd.date_range("01-01-2015 00:00:00", periods=14 * 24, freq="1h"),
            24,
        ),
        (
            DayOfWeek(normalized=False),
            pd.date_range("01-01-2015", periods=365 * 5, freq="D"),
            7,
        ),
        (
            DayOfMonth(normalized=False),
            pd.date_range("01-01-2015", periods=365 * 5, freq="D"),
            31,
        ),
        (
            DayOfYear(normalized=False),
            pd.date_range("01-01-2015", periods=365 * 5, freq="D"),
            366,
        ),
        (
            WeekOfYear(normalized=False),
            pd.date_range("01-01-2015", periods=53 * 5, freq="W"),
            53,
        ),
        (
            MonthOfYear(normalized=False),
            pd.date_range("01-01-2015", periods=12 * 5, freq="M"),
            12,
        ),
    ],
)
def test_feature_unnormalized_bounds(
    feature: TimeFeature, index: pd.DatetimeIndex, cardinality: int
):
    values = feature(index)
    counts = [0] * cardinality
    for v in values:
        assert 0 <= int(v) < cardinality
        counts[int(v)] += 1
    assert all(c > 0 for c in counts)
