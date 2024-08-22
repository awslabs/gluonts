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
from pandas.tseries.frequencies import to_offset

from gluonts.time_feature import norm_freq_str

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
