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

from typing import Callable, Dict, List

import numpy as np
from pydantic import BaseModel

import gluonts.zebras as zb

TimeFeature = Callable[[zb.Periods], np.ndarray]


def second_of_minute(index: zb.Periods) -> np.ndarray:
    """
    Second of minute encoded as value between [-0.5, 0.5]
    """
    return index.second / 59.0 - 0.5


def second_of_minute_index(index: zb.Periods) -> np.ndarray:
    """
    Second of minute encoded as zero-based index, between 0 and 59.
    """
    return index.second.astype(float)


def minute_of_hour(index: zb.Periods) -> np.ndarray:
    """
    Minute of hour encoded as value between [-0.5, 0.5]
    """
    return index.minute / 59.0 - 0.5


def minute_of_hour_index(index: zb.Periods) -> np.ndarray:
    """
    Minute of hour encoded as zero-based index, between 0 and 59.
    """

    return index.minute.astype(float)


def hour_of_day(index: zb.Periods) -> np.ndarray:
    """
    Hour of day encoded as value between [-0.5, 0.5]
    """

    return index.hour / 23.0 - 0.5


def hour_of_day_index(index: zb.Periods) -> np.ndarray:
    """
    Hour of day encoded as zero-based index, between 0 and 23.
    """

    return index.hour.astype(float)


def day_of_week(index: zb.Periods) -> np.ndarray:
    """
    Day of week encoded as value between [-0.5, 0.5]
    """

    return index.dayofweek / 6.0 - 0.5


def day_of_week_index(index: zb.Periods) -> np.ndarray:
    """
    Day of week encoded as zero-based index, between 0 and 6.
    """

    return index.dayofweek.astype(float)


def day_of_month(index: zb.Periods) -> np.ndarray:
    """
    Day of month encoded as value between [-0.5, 0.5]
    """

    return (index.day - 1) / 30.0 - 0.5


def day_of_month_index(index: zb.Periods) -> np.ndarray:
    """
    Day of month encoded as zero-based index, between 0 and 11.
    """

    return index.day.astype(float) - 1


def day_of_year(index: zb.Periods) -> np.ndarray:
    """
    Day of year encoded as value between [-0.5, 0.5]
    """

    return (index.dayofyear - 1) / 365.0 - 0.5


def day_of_year_index(index: zb.Periods) -> np.ndarray:
    """
    Day of year encoded as zero-based index, between 0 and 365.
    """

    return index.dayofyear.astype(float) - 1


def month_of_year(index: zb.Periods) -> np.ndarray:
    """
    Month of year encoded as value between [-0.5, 0.5]
    """

    return (index.month - 1) / 11.0 - 0.5


def month_of_year_index(index: zb.Periods) -> np.ndarray:
    """
    Month of year encoded as zero-based index, between 0 and 11.
    """

    return index.month.astype(float) - 1


def week_of_year(index: zb.Periods) -> np.ndarray:
    """
    Week of year encoded as value between [-0.5, 0.5]
    """
    return week_of_year_index(index) / 52.0 - 0.5


def week_of_year_index(index: zb.Periods) -> np.ndarray:
    """
    Week of year encoded as zero-based index, between 0 and 52.
    """
    return index.week - 1


class Constant(BaseModel):
    """
    Constant time feature using a predefined value.
    """

    value: float = 0.0

    def __call__(self, index: zb.Periods) -> np.ndarray:
        return np.full(index.data.shape, self.value)


def norm_freq_str(freq_str: str) -> str:
    base_freq = freq_str.split("-")[0]

    # Pandas has start and end frequencies, e.g `AS` and `A` for yearly start
    # and yearly end frequencies. We don't make that difference and instead
    # rely only on the end frequencies which don't have the `S` prefix.
    # Note: Secondly ("S") frequency exists, where we don't want to remove the
    # "S"!
    if len(base_freq) >= 2 and base_freq.endswith("S"):
        return base_freq[:-1]

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

    features_by_offsets: Dict[str, List[TimeFeature]] = {
        "Y": [],
        "Q": [month_of_year],
        "M": [month_of_year],
        "W": [day_of_month, week_of_year],
        "D": [day_of_week, day_of_month, day_of_year],
        # offsets.BusinessDay: [day_of_week, day_of_month, day_of_year],
        "h": [hour_of_day, day_of_week, day_of_month, day_of_year],
        "m": [
            minute_of_hour,
            hour_of_day,
            day_of_week,
            day_of_month,
            day_of_year,
        ],
        "s": [
            second_of_minute,
            minute_of_hour,
            hour_of_day,
            day_of_week,
            day_of_month,
            day_of_year,
        ],
    }

    freq = zb.Freq.from_pandas(freq_str)

    if freq.np_freq == ("M", 3):
        return features_by_offsets["Q"]

    try:
        return features_by_offsets[freq.np_freq[0]]
    except KeyError:
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
