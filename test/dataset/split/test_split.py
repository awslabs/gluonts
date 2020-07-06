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

from gluonts.dataset.split.splitter import TimeSeriesSlice


def make_series(data, start="2020", freq="D"):
    index = pd.date_range(start=start, freq=freq, periods=len(data))
    return pd.Series(data, index=index)


def test_ts_slice_to_item():

    sl = TimeSeriesSlice(
        target=make_series(range(100)),
        item="",
        feat_static_cat=[1, 2, 3],
        feat_static_real=[0.1, 0.2, 0.3],
        feat_dynamic_cat=[make_series(range(100))],
        feat_dynamic_real=[make_series(range(100))],
    )

    sl.to_data_entry()
