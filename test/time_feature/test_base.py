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

import pytest
from pandas.tseries import offsets

from gluonts.time_feature import norm_freq_str, to_offset

from .common import M, Q, S, Y


@pytest.mark.parametrize(
    " aliases, normalized_freq_str",
    [
        (["Y", "YS", "A", "AS"], Y),
        (["Q", "QS"], Q),
        (["M", "MS"], M),
        (["S"], S),
    ],
)
def test_norm_freq_str(aliases, normalized_freq_str):
    for alias in aliases:
        assert norm_freq_str(to_offset(alias).name) == normalized_freq_str


@pytest.mark.parametrize(
    "freq, expected",
    [
        # period aliases, which pandas >= 3 rejects for offsets
        ("M", offsets.MonthEnd()),
        ("3M", offsets.MonthEnd(3)),
        ("Q", offsets.QuarterEnd(startingMonth=12)),
        ("Q-NOV", offsets.QuarterEnd(startingMonth=11)),
        ("Y", offsets.YearEnd(month=12)),
        # legacy aliases, which pandas >= 3 rejects
        ("H", offsets.Hour()),
        ("2H", offsets.Hour(2)),
        ("1 H", offsets.Hour()),
        ("30T", offsets.Minute(30)),
        ("S", offsets.Second()),
        ("A", offsets.YearEnd(month=12)),
        ("A-DEC", offsets.YearEnd(month=12)),
        ("AS", offsets.YearBegin(month=1)),
        # aliases accepted by all supported pandas versions
        ("h", offsets.Hour()),
        ("5min", offsets.Minute(5)),
        ("D", offsets.Day()),
        ("W", offsets.Week(weekday=6)),
        ("B", offsets.BusinessDay()),
        (offsets.Hour(2), offsets.Hour(2)),
    ],
)
def test_to_offset(freq, expected):
    assert to_offset(freq) == expected


def test_to_offset_invalid():
    with pytest.raises(ValueError, match="XYZ"):
        to_offset("XYZ")
