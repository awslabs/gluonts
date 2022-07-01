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

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.split import DateSplitter, OffsetSplitter, split
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
    original_entry,
    train_entry,
    valid_pair,
    prediction_length: int,
    max_history,
    date,
    offset,
) -> None:
    assert (
        str(original_entry[FieldName.ITEM_ID])
        == train_entry[FieldName.ITEM_ID]
    )
    assert train_entry[FieldName.ITEM_ID] == valid_pair[0][FieldName.ITEM_ID]
    if max_history is not None:
        assert len(valid_pair[0][FieldName.TARGET]) == max_history
    assert len(valid_pair[1][FieldName.TARGET]) == prediction_length
    train_end = (
        train_entry[FieldName.START]
        + len(train_entry[FieldName.TARGET])
        * train_entry[FieldName.START].freq
    )
    if date is not None:
        assert train_end == date + train_entry[FieldName.START].freq
    if offset is not None:
        if offset > 0:
            assert len(train_entry[FieldName.TARGET]) == offset
        else:
            assert len(train_entry[FieldName.TARGET]) - offset == len(
                original_entry[FieldName.TARGET]
            )
    assert train_end <= valid_pair[1][FieldName.START]
    valid_end = (
        valid_pair[0][FieldName.START]
        + len(valid_pair[0][FieldName.TARGET])
        * valid_pair[0][FieldName.START].freq
    )
    assert valid_end == valid_pair[1][FieldName.START]


def test_split_mult_freq():
    splitter = DateSplitter(
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

    splitter = OffsetSplitter(split_offset=-7).split(dataset)

    assert [len(t["target"]) for t in splitter[0]] == [93, 43]
    assert [
        len(t["target"]) + len(s["target"])
        for t, s in splitter[1].generate_instances(prediction_length=7)
    ] == [100, 50]

    rolling_splitter = OffsetSplitter(split_offset=-21).split(dataset)

    assert [len(t["target"]) for t in rolling_splitter[0]] == [79, 29]
    assert [
        len(t["target"]) + len(s["target"])
        for t, s in rolling_splitter[1].generate_instances(
            prediction_length=7, windows=3
        )
    ] == [
        86,
        93,
        100,
        36,
        43,
        50,
    ]


@pytest.mark.parametrize(
    "date, offset, windows, distance, max_history",
    [
        (pd.Period("2021-06-17", freq="D"), None, 1, None, None),
        (pd.Period("2021-06-17", freq="D"), None, 1, None, 96),
        (pd.Period("2021-06-17", freq="D"), None, 3, None, None),
        (pd.Period("2021-06-17", freq="D"), None, 3, None, 96),
        (None, 192, 1, None, 96),
        (None, 192, 3, None, 96),
        (None, -48, 1, None, None),
    ],
)
def test_split(date, offset, windows, distance, max_history):
    assert (offset is None) != (date is None)

    dataset = ListDataset(
        [
            {"item_id": 0, "start": "2021-03-04", "target": [1.0] * 365},
            {"item_id": 1, "start": "2021-03-04", "target": [2.0] * 265},
        ],
        freq="D",
    )

    prediction_length = 7

    train, test_template = split(dataset, date=date, offset=offset)
    train = list(train)

    validation = list(
        test_template.generate_instances(
            prediction_length=prediction_length,
            windows=windows,
            distance=distance,
            max_history=max_history,
        )
    )

    assert len(train) == len(dataset)
    assert len(validation) == windows * len(train)

    k = 0
    for original_entry, train_entry in zip(dataset, train):
        for _ in range(windows):
            valid_pair = validation[k]
            check_training_validation(
                original_entry=original_entry,
                train_entry=train_entry,
                valid_pair=valid_pair,
                prediction_length=prediction_length,
                max_history=max_history,
                date=date,
                offset=offset,
            )
            k += 1
