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

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.split import DateSplitter, OffsetSplitter
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


def test_splitter():

    dataset = get_dataset("m4_hourly")
    prediction_length = dataset.metadata.prediction_length
    splitter = DateSplitter(
        prediction_length=prediction_length,
        split_date=pd.Timestamp("1750-01-05 04:00:00", freq="h"),
    )
    train, validation = splitter.split(dataset.train)
    assert len(train[1][0][FieldName.TARGET]) + prediction_length == len(
        validation[1][0][FieldName.TARGET]
    )

    max_history = 2 * prediction_length
    splitter = OffsetSplitter(
        prediction_length=prediction_length,
        split_offset=4 * prediction_length,
        max_history=max_history,
    )
    train, validation = splitter.split(dataset.train)
    assert len(validation[1][0][FieldName.TARGET]) == max_history
    assert len(train[1][0][FieldName.TARGET]) == 4 * prediction_length

    train, validation = splitter.rolling_split(dataset.train, windows=3)
    for i in range(3):
        assert len(validation[1][i][FieldName.TARGET]) == max_history
        assert len(train[1][i][FieldName.TARGET]) == 4 * prediction_length

    max_history = 2 * prediction_length
    splitter = DateSplitter(
        prediction_length=prediction_length,
        split_date=pd.Timestamp("1750-01-05 04:00:00", freq="h"),
        max_history=max_history,
    )
    train, validation = splitter.split(dataset.train)
    assert len(validation[1][0][FieldName.TARGET]) == max_history

    train, validation = splitter.rolling_split(dataset.train, windows=3)
    for i in range(3):
        assert len(validation[1][i][FieldName.TARGET]) == max_history
