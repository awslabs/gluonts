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

from gluonts.time_feature import get_seasonality

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
