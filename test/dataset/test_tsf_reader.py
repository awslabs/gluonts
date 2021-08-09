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

from gluonts.dataset.repository._tsf_reader import frequency_converter


@pytest.mark.parametrize(
    "input_freq_str, output_freq_str",
    [
        ("hourly", "H"),
        ("minutely", "T"),
        ("daily", "D"),
        ("half_hourly", "0.5H"),
        ("weekly", "W"),
        ("monthly", "M"),
        ("quarterly", "Q"),
        ("30_seconds", "30S"),
        ("10_minutes", "10T"),
        ("4_weeks", "4W"),
        ("2_months", "2M"),
    ],
)
def test_frequency_converter(input_freq_str: str, output_freq_str: str):
    assert frequency_converter(input_freq_str) == output_freq_str
