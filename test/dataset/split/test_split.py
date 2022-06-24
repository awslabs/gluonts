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
    index = pd.period_range(start=start, freq=freq, periods=len(data))
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


def check_training_validation(
    train_entry,
    valid_pair,
    prediction_length: int,
    cutoff_date=None,
    max_history=None,
    split_offset=None,
    window=1,
    is_rolling_split=False,
) -> None:
    assert train_entry[FieldName.ITEM_ID] == valid_pair[0][FieldName.ITEM_ID]
    if max_history is None:
        assert len(train_entry[FieldName.TARGET]) == len(
            valid_pair[0][FieldName.TARGET]
        )
    elif not is_rolling_split:
        assert (
            len(valid_pair[0][FieldName.TARGET]) + len(valid_pair[1])
            == max_history
        )
    assert len(valid_pair[1]) == prediction_length * window
    train_end = (
        train_entry[FieldName.START]
        + len(train_entry[FieldName.TARGET])
        * train_entry[FieldName.START].freq
    )
    if cutoff_date:
        assert train_end == cutoff_date + train_entry[FieldName.START].freq
    if split_offset:
        assert len(train_entry[FieldName.TARGET]) == split_offset
    assert train_end <= valid_pair[1].index[0]
    valid_end = (
        valid_pair[0][FieldName.START]
        + len(valid_pair[0][FieldName.TARGET])
        * valid_pair[0][FieldName.START].freq
    )
    assert valid_end == valid_pair[1].index[0]


def test_splitter():
    dataset = get_dataset("m4_hourly")
    prediction_length = dataset.metadata.prediction_length
    split_date = pd.Period("1750-01-05 04:00:00", freq="h")
    splitter = DateSplitter(
        prediction_length=prediction_length,
        split_date=split_date,
    )
    train, validation = splitter.split(dataset.train)
    for train_entry, valid_pair in zip(train, validation):
        check_training_validation(
            train_entry, valid_pair, prediction_length, cutoff_date=split_date
        )
    assert len(list(train)) == len(list(validation))

    max_history = 2 * prediction_length
    split_offset = 4 * prediction_length
    splitter = OffsetSplitter(
        prediction_length=prediction_length,
        split_offset=split_offset,
        max_history=max_history,
    )
    train, validation = splitter.split(dataset.train)
    for train_entry, valid_pair in zip(train, validation):
        check_training_validation(
            train_entry,
            valid_pair,
            prediction_length,
            max_history=max_history,
            split_offset=split_offset,
        )
    assert len(list(train)) == len(list(validation))

    windows = 3
    splitter = OffsetSplitter(
        prediction_length=prediction_length,
        split_offset=split_offset,
        max_history=max_history,
        windows=windows,
    )
    train, validation = splitter.split(dataset.train)
    valid_list = list(validation)
    train_list = list(train)
    k = 0
    for train_entry in train_list:
        for i in range(windows):
            valid_pair = valid_list[k]
            window = i + 1
            check_training_validation(
                train_entry,
                valid_pair,
                prediction_length,
                max_history=max_history,
                split_offset=split_offset,
                window=window,
                is_rolling_split=True,
            )
            k += 1

    max_history = 2 * prediction_length
    splitter = DateSplitter(
        prediction_length=prediction_length,
        split_date=split_date,
        max_history=max_history,
    )
    train, validation = splitter.split(dataset.train)
    for train_entry, valid_pair in zip(train, validation):
        check_training_validation(
            train_entry,
            valid_pair,
            prediction_length,
            max_history=max_history,
            cutoff_date=split_date,
        )
    assert len(list(train)) == len(list(validation))

    splitter = DateSplitter(
        prediction_length=prediction_length,
        split_date=split_date,
        max_history=max_history,
        windows=windows,
    )
    train, validation = splitter.split(dataset.train)
    valid_list = list(validation)
    train_list = list(train)
    k = 0
    for train_entry in train_list:
        for i in range(windows):
            valid_pair = valid_list[k]
            window = i + 1
            check_training_validation(
                train_entry,
                valid_pair,
                prediction_length,
                max_history=max_history,
                cutoff_date=split_date,
                window=window,
                is_rolling_split=True,
            )
            k += 1


def test_split_mult_freq():
    splitter = DateSplitter(
        prediction_length=1,
        split_date=pd.Period("2021-01-01", "2h"),
    )

    splitter.split(
        [
            {
                "item_id": "1",
                "target": pd.Series([0, 1, 2]),
                "start": pd.Period("2021-01-01", freq="2H"),
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
