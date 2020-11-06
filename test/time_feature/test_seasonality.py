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


@pytest.mark.parametrize(
    "freq, expected_seasonality",
    [
        ("1H", 24),
        ("H", 24),
        ("2H", 12),
        ("3H", 8),
        ("4H", 6),
        ("15H", 1),
        ("5B", 1),
        ("1B", 5),
        ("2W", 1),
        ("3M", 4),
        ("1D", 1),
        ("7D", 1),
        ("8D", 1),
    ],
)
def test_get_seasonality(freq, expected_seasonality):
    assert get_seasonality(freq) == expected_seasonality
