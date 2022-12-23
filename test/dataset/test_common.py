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

import pandas as pd
import pytest
from gluonts.dataset.common import Dataset


@pytest.mark.parametrize(
    "freq, expected, given",
    [
        ("B", "2019-11-01", "2019-11-01 12:34:56"),
        ("W", "2019-11-03", "2019-11-01 12:34:56"),
        ("M", "2019-11-30", "2019-11-01 12:34:56"),
        ("12M", "2019-11-30", "2019-11-01 12:34:56"),
        ("A-DEC", "2019-12-31", "2019-11-01 12:34:56"),
    ],
)
def test_process_start_field(freq, expected, given):
    assert pd.Period(given, freq) == pd.Period(expected, freq)


def test_dataset_instance():
    assert isinstance([], Dataset)
