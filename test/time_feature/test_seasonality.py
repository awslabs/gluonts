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

import warnings

import pytest

from gluonts.time_feature import get_seasonality
from gluonts.time_feature.seasonality import get_seasonality_for_frequency

from .common import H, M, Q, Y

TEST_CASES = [
    ("30min", 48),
    ("5B", 1),
    ("1B", 5),
    ("2W", 1),
    ("1D", 1),
    ("7D", 1),
    ("8D", 1),
    # Monthly
    ("MS", 12),
    ("3MS", 4),
    (M, 12),
    ("3" + M, 4),
    # Quarterly
    ("QS", 4),
    ("2QS", 2),
    (Q, 4),
    ("2" + Q, 2),
    ("3" + Q, 1),
    # Hourly
    ("1" + H, 24),
    (H, 24),
    ("2" + H, 12),
    ("3" + H, 8),
    ("4" + H, 6),
    ("15" + H, 1),
    # Yearly
    (Y, 1),
    ("2" + Y, 1),
    ("YS", 1),
    ("2YS", 1),
]


@pytest.mark.parametrize("freq, expected_seasonality", TEST_CASES)
def test_get_seasonality(freq, expected_seasonality):
    assert get_seasonality(freq) == expected_seasonality


def test_get_seasonality_for_frequency():
    """The new canonical function should return the same values."""
    assert get_seasonality_for_frequency("H") == 24
    assert get_seasonality_for_frequency("2H") == 12
    assert get_seasonality_for_frequency("30min") == 48
    assert get_seasonality_for_frequency("D") == 1
    assert get_seasonality_for_frequency("W") == 1
    assert get_seasonality_for_frequency("M") == 12
    assert get_seasonality_for_frequency("3M") == 4
    assert get_seasonality_for_frequency("1B") == 5


def test_get_seasonality_deprecation_warning():
    """The old get_seasonality should emit a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        get_seasonality("H")
        deprecation_warnings = [
            ww for ww in w
            if issubclass(ww.category, DeprecationWarning)
            and "get_seasonality_for_frequency" in str(ww.message)
        ]
        assert len(deprecation_warnings) == 1
