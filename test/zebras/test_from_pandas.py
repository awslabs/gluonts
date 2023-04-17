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

import gluonts.zebras as zb


@pytest.mark.parametrize("from_pandas", [zb.from_pandas, zb.Freq.from_pandas])
@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("freq", ["H", "D", "W"])
def test_pandas_freq(from_pandas, n, freq):
    pd_freq = to_offset(f"{n} {freq}")
    assert to_offset(str(from_pandas(pd_freq))) == pd_freq


@pytest.mark.parametrize(
    "from_pandas", [zb.from_pandas, zb.TimeFrame.from_pandas]
)
def test_pandas_dataframe(from_pandas):
    df = pd.DataFrame(
        {"x": [1, 2, 3]},
        index=pd.date_range(start="2020", periods=3, freq="H"),
    )

    tf = zb.TimeFrame.from_pandas(df)
    assert np.array_equal(tf.index.data.astype(int), df.index.to_period().asi8)
    assert np.array_equal(tf["x"], df["x"])

    df = df.reindex(pd.period_range(start="2020", periods=3, freq="H"))
    tf = zb.TimeFrame.from_pandas(df)
    assert np.array_equal(tf.index.data.astype(int), df.index.asi8)

    df = df.reindex(np.arange(len(df)))
    tf = zb.TimeFrame.from_pandas(df)
    assert tf.index is None
