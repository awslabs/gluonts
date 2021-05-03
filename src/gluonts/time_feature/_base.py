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
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

from gluonts.core.component import validated


class TimeFeature:
    """
    Base class for features that only depend on time.
    """

    @validated()
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class MinuteOfHourIndex(TimeFeature):
    """Minute of hour encoded as zero-based index, between 0 and 59"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute.map(float)


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class HourOfDayIndex(TimeFeature):
    """Hour of day encoded as zero-based index, between 0 and 23"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour.map(float)


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfWeekIndex(TimeFeature):
    """Hour of day encoded as zero-based index, between 0 and 6"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek.map(float)


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfMonthIndex(TimeFeature):
    """Day of month encoded as zero-based index, between 0 and 11"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1).map(float)


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class DayOfYearIndex(TimeFeature):
    """Day of year encoded as zero-based index, between 0 and 365"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1).map(float)


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class MonthOfYearIndex(TimeFeature):
    """Month of year encoded as zero-based index, between 0 and 11"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1).map(float)


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # TODO:
        # * pandas >= 1.1 does not support `.week`
        # * pandas == 1.0 does not support `.isocalendar()`
        # as soon as we drop support for `pandas == 1.0`, we should remove this
        try:
            week = index.isocalendar().week
        except AttributeError:
            week = index.week
        return (week - 1) / 52.0 - 0.5


class WeekOfYearIndex(TimeFeature):
    """Week of year encoded as zero-based index, between 0 and 52"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # TODO:
        # * pandas >= 1.1 does not support `.week`
        # * pandas == 1.0 does not support `.isocalendar()`
        # as soon as we drop support for `pandas == 1.0`, we should remove this
        try:
            week = index.isocalendar().week
        except AttributeError:
            week = index.week
        return (week - 1).map(float)


def norm_freq_str(freq_str: str) -> str:
    return freq_str.split("-")[0]


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.

    Parameters
    ----------

    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}

    The following frequencies are supported:

        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
    """
    raise RuntimeError(supported_freq_msg)
