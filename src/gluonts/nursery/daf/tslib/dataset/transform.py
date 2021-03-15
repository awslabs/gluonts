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


from typing import Optional
from copy import copy, deepcopy

import numpy as np
import pandas as pd

from .timeseries import TimeSeries


def add_datetime_features(
    ts: TimeSeries,
    add_month_of_year: bool = False,
    add_week_of_year: bool = False,
    add_day_of_year: bool = False,
    add_day_of_month: bool = False,
    add_day_of_week: bool = False,
    add_hour_of_day: bool = False,
    add_minute_of_hour: bool = False,
    normalized: bool = True,
) -> TimeSeries:
    def _month_of_year(time_index: pd.DatetimeIndex) -> np.ndarray:
        vals = time_index.month.values - 1
        if normalized:
            vals = vals.astype(np.float32)
            vals = vals / 11.0 - 0.5
            vals = vals.reshape(-1, 1)
            return vals
        else:
            return vals.astype(np.int64)

    def _week_of_year(time_index: pd.DatetimeIndex) -> np.ndarray:
        vals = time_index.weekofyear.values - 1
        if normalized:
            vals = vals.astype(np.float32)
            vals = vals / 51.0 - 0.5
            vals = vals.reshape(-1, 1)
            return vals
        else:
            return vals.astype(np.int64)

    def _day_of_year(time_index: pd.DatetimeIndex) -> np.ndarray:
        vals = time_index.dayofyear.values - 1
        if normalized:
            vals = vals.astype(np.float32)
            vals = vals / 364.0 - 0.5
            vals = vals.reshape(-1, 1)
            return vals
        else:
            return vals.astype(np.int64)

    def _day_of_month(time_index: pd.DatetimeIndex) -> np.ndarray:
        vals = time_index.day.values - 1
        if normalized:
            vals = vals.astype(np.float32)
            vals = vals / 30.0 - 0.5
            vals = vals.reshape(-1, 1)
            return vals
        else:
            return vals.astype(np.int64)

    def _day_of_week(time_index: pd.DatetimeIndex) -> np.ndarray:
        vals = time_index.dayofweek.values
        if normalized:
            vals = vals.astype(np.float32)
            vals = vals / 6.0 - 0.5
            vals = vals.reshape(-1, 1)
            return vals
        else:
            return vals.astype(np.int64)

    def _hour_of_day(time_index: pd.DatetimeIndex) -> np.ndarray:
        vals = time_index.hour.values - 1
        if normalized:
            vals = vals.astype(np.float32)
            vals = vals / 23.0 - 0.5
            vals = vals.reshape(-1, 1)
            return vals
        else:
            return vals.astype(np.int64)

    def _minute_of_hour(time_index: pd.DatetimeIndex) -> np.ndarray:
        vals = time_index.minute.values - 1
        if normalized:
            vals = vals.astype(np.float32)
            vals = vals / 59.0 - 0.5
            vals = vals.reshape(-1, 1)
            return vals
        else:
            return vals.astype(np.int64)

    time_index = ts.time_index
    feature_type = "revealed"
    data_type = "numerical" if normalized else "categorical"
    if add_month_of_year:
        ts.add_features(
            "month_of_year",
            _month_of_year(time_index),
            data_type,
            feature_type,
        )
    if add_week_of_year:
        ts.add_features(
            "week_of_year", _week_of_year(time_index), data_type, feature_type
        )
    if add_day_of_year:
        ts.add_features(
            "day_of_year", _day_of_year(time_index), data_type, feature_type
        )
    if add_day_of_month:
        ts.add_features(
            "day_of_month", _day_of_month(time_index), data_type, feature_type
        )
    if add_day_of_week:
        ts.add_features(
            "day_of_week", _day_of_week(time_index), data_type, feature_type
        )
    if add_hour_of_day:
        ts.add_features(
            "hour_of_day", _hour_of_day(time_index), data_type, feature_type
        )
    if add_minute_of_hour:
        ts.add_features(
            "minute_of_hour",
            _minute_of_hour(time_index),
            data_type,
            feature_type,
        )
    return ts


def add_duration_feature(
    ts: TimeSeries, log_scale: Optional[float] = None, normalized: bool = True
) -> TimeSeries:
    freq = pd.infer_freq(ts.time_index)
    if freq is None:
        age = np.arange(len(ts), dtype=np.float)
    else:
        age = ts.time_index - ts.time_index[0]
        if freq == "H":
            age = age.seconds.values // 3600
        elif freq == "D":
            age = age.days.values
        else:
            raise NotImplementedError
    if log_scale is not None:
        age = np.log1p(age) / np.log(log_scale)
    if normalized:
        age = age / np.max(age) - 0.5
    ts.add_features(
        "age",
        age.reshape(-1, 1),
        data_type="numerical",
        feature_type="revealed",
    )
    return ts
