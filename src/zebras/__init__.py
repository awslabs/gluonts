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

__all__ = [
    "Freq",
    "freq",
    "Period",
    "Periods",
    "period",
    "periods",
    "TimeFrame",
    "time_frame",
    "SplitFrame",
    "split_frame",
    "time_series",
    "TimeSeries",
]

from ._freq import Freq, freq
from ._period import period, Period, periods, Periods
from ._split_frame import split_frame, SplitFrame
from ._timeframe import time_frame, TimeFrame
from ._time_series import time_series, TimeSeries


def batch(xs: list):
    assert xs

    return xs[0]._batch(xs)
