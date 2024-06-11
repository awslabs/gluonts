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

from packaging.version import Version
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

from gluonts.pydantic import BaseModel


TimeFeature = Callable[[pd.PeriodIndex], np.ndarray]


def _normalize(xs, num: float):
    """
    Scale values of ``xs`` to [-0.5, 0.5].
    """

    return np.asarray(xs) / (num - 1) - 0.5


def second_of_minute(index: pd.PeriodIndex) -> np.ndarray:
    """
    Second of minute encoded as value between [-0.5, 0.5]
    """
    return _normalize(index.second, num=60)


def second_of_minute_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Second of minute encoded as zero-based index, between 0 and 59.
    """
    return np.asarray(index.second)


def minute_of_hour(index: pd.PeriodIndex) -> np.ndarray:
    """
    Minute of hour encoded as value between [-0.5, 0.5]
    """
    return _normalize(index.minute, num=60)


def minute_of_hour_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Minute of hour encoded as zero-based index, between 0 and 59.
    """

    return np.asarray(index.minute)


def hour_of_day(index: pd.PeriodIndex) -> np.ndarray:
    """
    Hour of day encoded as value between [-0.5, 0.5]
    """

    return _normalize(index.hour, num=24)


def hour_of_day_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Hour of day encoded as zero-based index, between 0 and 23.
    """

    return np.asarray(index.hour)


def day_of_week(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of week encoded as value between [-0.5, 0.5]
    """

    return _normalize(index.dayofweek, num=7)


def day_of_week_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of week encoded as zero-based index, between 0 and 6.
    """

    return np.asarray(index.dayofweek)


def day_of_month(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of month encoded as value between [-0.5, 0.5]
    """

    # first day of month is `1`, thus we deduct one
    return _normalize(index.day - 1, num=31)


def day_of_month_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of month encoded as zero-based index, between 0 and 11.
    """

    return np.asarray(index.day) - 1


def day_of_year(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of year encoded as value between [-0.5, 0.5]
    """

    return _normalize(index.dayofyear - 1, num=366)


def day_of_year_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of year encoded as zero-based index, between 0 and 365.
    """

    return np.asarray(index.dayofyear) - 1


def month_of_year(index: pd.PeriodIndex) -> np.ndarray:
    """
    Month of year encoded as value between [-0.5, 0.5]
    """

    return _normalize(index.month - 1, num=12)


def month_of_year_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Month of year encoded as zero-based index, between 0 and 11.
    """

    return np.asarray(index.month) - 1


def week_of_year(index: pd.PeriodIndex) -> np.ndarray:
    """
    Week of year encoded as value between [-0.5, 0.5]
    """

    # TODO:
    # * pandas >= 1.1 does not support `.week`
    # * pandas == 1.0 does not support `.isocalendar()`
    # as soon as we drop support for `pandas == 1.0`, we should remove this
    try:
        week = index.isocalendar().week
    except AttributeError:
        week = index.week

    return _normalize(week - 1, num=53)


def week_of_year_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Week of year encoded as zero-based index, between 0 and 52.
    """

    # TODO:
    # * pandas >= 1.1 does not support `.week`
    # * pandas == 1.0 does not support `.isocalendar()`
    # as soon as we drop support for `pandas == 1.0`, we should remove this
    try:
        week = index.isocalendar().week
    except AttributeError:
        week = index.week

    return np.asarray(week) - 1


class Constant(BaseModel):
    """
    Constant time feature using a predefined value.
    """

    value: float = 0.0

    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.full(index.shape, self.value)


def norm_freq_str(freq_str: str) -> str:
    base_freq = freq_str.split("-")[0]

    # Pandas has start and end frequencies, e.g `AS` and `A` for yearly start
    # and yearly end frequencies. We don't make that difference and instead
    # rely only on the end frequencies which don't have the `S` prefix.
    # Note: Secondly ("S") frequency exists, where we don't want to remove the
    # "S"!
    if len(base_freq) >= 2 and base_freq.endswith("S"):
        base_freq = base_freq[:-1]
        # In pandas >= 2.2, period end frequencies have been renamed, e.g. "M" -> "ME"
        if Version(pd.__version__) >= Version("2.2.0"):
            base_freq += "E"

    return base_freq


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given
    frequency string.

    Parameters
    ----------

    freq_str
        Frequency string of the form [multiple][granularity] such as "12H",
        "5min", "1D" etc.
    """

    features_by_offsets: Dict[Any, List[TimeFeature]] = {
        offsets.YearBegin: [],
        offsets.YearEnd: [],
        offsets.QuarterBegin: [month_of_year],
        offsets.QuarterEnd: [month_of_year],
        offsets.MonthBegin: [month_of_year],
        offsets.MonthEnd: [month_of_year],
        offsets.Week: [day_of_month, week_of_year],
        offsets.Day: [day_of_week, day_of_month, day_of_year],
        offsets.BusinessDay: [day_of_week, day_of_month, day_of_year],
        offsets.Hour: [hour_of_day, day_of_week, day_of_month, day_of_year],
        offsets.Minute: [
            minute_of_hour,
            hour_of_day,
            day_of_week,
            day_of_month,
            day_of_year,
        ],
        offsets.Second: [
            second_of_minute,
            minute_of_hour,
            hour_of_day,
            day_of_week,
            day_of_month,
            day_of_year,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, features in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return features

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}

    The following frequencies are supported:
    
    """

    for offset_cls in features_by_offsets:
        offset = offset_cls()
        supported_freq_msg += (
            f"\t{offset.freqstr.split('-')[0]} - {offset_cls.__name__}"
        )

    raise RuntimeError(supported_freq_msg)
