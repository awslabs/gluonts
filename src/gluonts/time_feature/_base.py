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

from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from pydantic import BaseModel


TimeFeature = Callable[[pd.PeriodIndex], np.ndarray]


def second_of_minute(index: pd.PeriodIndex) -> np.ndarray:
    """
    Second of minute encoded as value between [-0.5, 0.5]
    """
    return index.second.values / 59.0 - 0.5


def second_of_minute_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Second of minute encoded as zero-based index, between 0 and 59.
    """
    return index.second.astype(float).values


def minute_of_hour(index: pd.PeriodIndex) -> np.ndarray:
    """
    Minute of hour encoded as value between [-0.5, 0.5]
    """
    return index.minute.values / 59.0 - 0.5


def minute_of_hour_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Minute of hour encoded as zero-based index, between 0 and 59.
    """

    return index.minute.astype(float).values


def hour_of_day(index: pd.PeriodIndex) -> np.ndarray:
    """
    Hour of day encoded as value between [-0.5, 0.5]
    """

    return index.hour.values / 23.0 - 0.5


def hour_of_day_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Hour of day encoded as zero-based index, between 0 and 23.
    """

    return index.hour.astype(float).values


def day_of_week(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of week encoded as value between [-0.5, 0.5]
    """

    return index.dayofweek.values / 6.0 - 0.5


def day_of_week_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of week encoded as zero-based index, between 0 and 6.
    """

    return index.dayofweek.astype(float).values


def day_of_month(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of month encoded as value between [-0.5, 0.5]
    """

    return (index.day.values - 1) / 30.0 - 0.5


def day_of_month_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of month encoded as zero-based index, between 0 and 11.
    """

    return index.day.astype(float).values - 1


def day_of_year(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of year encoded as value between [-0.5, 0.5]
    """

    return (index.dayofyear.values - 1) / 365.0 - 0.5


def day_of_year_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of year encoded as zero-based index, between 0 and 365.
    """

    return index.dayofyear.astype(float).values - 1


def month_of_year(index: pd.PeriodIndex) -> np.ndarray:
    """
    Month of year encoded as value between [-0.5, 0.5]
    """

    return (index.month.values - 1) / 11.0 - 0.5


def month_of_year_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Month of year encoded as zero-based index, between 0 and 11.
    """

    return index.month.astype(float).values - 1


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
    return (week.astype(float).values - 1) / 52.0 - 0.5


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
    return week.astype(float).values - 1


class Constant(BaseModel):
    """
    Constant time feature using a predefined value.
    """

    value: float = 0.0

    def __call__(self, index: pd.PeriodIndex) -> np.ndarray:
        return np.full(index.shape, self.value)


def norm_freq_str(freq_str: str) -> str:
    return freq_str.split("-")[0]


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

        Y   - yearly
            alias: A
        Q   - quarterly
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)
