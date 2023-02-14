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

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from gluonts import zebras as zb

FREQS = ["S", "min", "T", "H", "D", "B", "W", "M", "QS", "Q", "AS", "A", "Y"]


@pytest.mark.parametrize("freq", FREQS)
def test_freq_equal_pandas(freq):
    zb_freq = zb.freq(freq)
    pd_freq = to_offset(freq)

    assert zb_freq.to_pandas() == pd_freq

    freq = "7" + freq
    zb_freq = zb.freq(freq)
    pd_freq = to_offset(freq)

    assert zb_freq.to_pandas() == pd_freq


@pytest.mark.parametrize("freq", FREQS)
def test_periods_equal_pandas(freq):
    if freq != "S" and freq.endswith("S"):
        pytest.skip()

    pr = pd.period_range(start="2020", freq=freq, periods=20)
    ps = zb.periods("2020", freq, 20)

    np.testing.assert_array_equal(pr, ps.to_pandas())

    freq = "7" + freq
    pr = pd.period_range(start="2020", freq=freq, periods=20)
    ps = zb.periods("2020", freq, 20)

    np.testing.assert_array_equal(pr, ps.to_pandas())
