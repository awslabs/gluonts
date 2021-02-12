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

from ._base import (
    TimeFeature,
    DayOfMonth,
    DayOfWeek,
    DayOfYear,
    HourOfDay,
    MinuteOfHour,
    MonthOfYear,
    WeekOfYear,
    DayOfMonthIndex,
    DayOfWeekIndex,
    DayOfYearIndex,
    HourOfDayIndex,
    MinuteOfHourIndex,
    MonthOfYearIndex,
    WeekOfYearIndex,
    time_features_from_frequency_str,
    norm_freq_str,
)
from .holiday import SPECIAL_DATE_FEATURES, SpecialDateFeatureSet
from .lag import get_lags_for_frequency
from .seasonality import get_seasonality

__all__ = [
    "TimeFeature",
    "DayOfMonth",
    "DayOfWeek",
    "DayOfYear",
    "HourOfDay",
    "MinuteOfHour",
    "MonthOfYear",
    "WeekOfYear",
    "DayOfMonthIndex",
    "DayOfWeekIndex",
    "DayOfYearIndex",
    "HourOfDayIndex",
    "MinuteOfHourIndex",
    "MonthOfYearIndex",
    "WeekOfYearIndex",
    "SPECIAL_DATE_FEATURES",
    "SpecialDateFeatureSet",
    "get_lags_for_frequency",
    "time_features_from_frequency_str",
    "get_seasonality",
]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
