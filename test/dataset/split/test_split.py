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

from gluonts.dataset.common import ListDataset
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
        split_date=pd.Timestamp("1750-01-05 04:00:00"),
    )
    train, validation = splitter.split(dataset.train)
    train = iter(train)
    validation = iter(validation)
    assert len(next(train)[FieldName.TARGET]) == len(
        next(validation)[0][FieldName.TARGET]
    )

    max_history = 2 * prediction_length
    splitter = OffsetSplitter(
        prediction_length=prediction_length,
        split_offset=4 * prediction_length,
        max_history=max_history,
    )
    train, validation = splitter.split(dataset.train)
    train = iter(train)
    validation = iter(validation)
    a = next(validation)
    assert len(a[0][FieldName.TARGET]) + len(a[1]) == max_history
    assert len(next(train)[FieldName.TARGET]) == 4 * prediction_length

    splitter = OffsetSplitter(
        prediction_length=prediction_length,
        split_offset=4 * prediction_length,
        max_history=max_history,
        windows=3,
    )
    train, validation = splitter.split(dataset.train)
    train = iter(train)
    validation = iter(validation)
    a = next(validation)
    for i in range(3):
        assert len(a[0][FieldName.TARGET]) + len(a[1]) == max_history
        assert len(next(train)[FieldName.TARGET]) == 4 * prediction_length

    max_history = 2 * prediction_length
    splitter = DateSplitter(
        prediction_length=prediction_length,
        split_date=pd.Timestamp("1750-01-05 04:00:00"),
        max_history=max_history,
    )
    train, validation = splitter.split(dataset.train)
    train = iter(train)
    validation = iter(validation)
    a = next(validation)
    assert len(a[0][FieldName.TARGET]) + len(a[1]) == max_history

    splitter = DateSplitter(
        prediction_length=prediction_length,
        split_date=pd.Timestamp("1750-01-05 04:00:00"),
        max_history=max_history,
        windows=3,
    )
    train, validation = splitter.split(dataset.train)
    train = iter(train)
    validation = iter(validation)
    a = next(validation)
    for i in range(3):
        assert len(a[0][FieldName.TARGET]) + len(a[1]) == max_history


def test_split_mult_freq():
    splitter = DateSplitter(
        prediction_length=1, split_date=pd.Timestamp("2021-01-01")
    )

    splitter.split(
        [
            {
                "item_id": "1",
                "target": pd.Series([0, 1, 2]),
                "start": pd.Timestamp("2021-01-01", freq="2H"),
            }
        ]
    )


def test_negative_offset_splitter():
    dataset = ListDataset(
        [
            {"item_id": 0, "start": "2021-03-04", "target": [1.0] * 100},
            {"item_id": 1, "start": "2021-03-04", "target": [2.0] * 50},
        ],
        freq="D",
    )

    split = OffsetSplitter(prediction_length=7, split_offset=-7).split(dataset)

    assert [len(t["target"]) for t in split[0]] == [93, 43]
    assert [len(t["target"]) + len(s) for t, s in split[1]] == [100, 50]

    rolling_split = OffsetSplitter(
        prediction_length=7, split_offset=-21, windows=3
    ).split(dataset)

    assert [len(t["target"]) for t in rolling_split[0]] == [79, 29]
    assert [len(t["target"]) + len(s) for t, s in rolling_split[1]] == [
        86,
        93,
        100,
        36,
        43,
        50,
    ]