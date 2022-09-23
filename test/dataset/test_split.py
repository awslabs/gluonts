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

import numpy as np
import pandas as pd
import pytest

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import (
    DateSplitter,
    OffsetSplitter,
    split,
    TimeSeriesSlice,
)


def test_time_series_slice():
    entry = {
        "start": pd.Period("2021-02-03", "D"),
        "target": np.array(range(100), dtype=float),
        "feat_dynamic_real": np.expand_dims(
            np.array(range(100), dtype=float), 0
        ),
    }

    tss = TimeSeriesSlice(entry)

    entry_slice = tss[10:20]
    assert entry_slice["start"] == pd.Period("2021-02-03", "D") + 10
    assert (entry_slice["target"] == np.arange(10, 20)).all()
    assert (
        entry_slice["feat_dynamic_real"] == np.array([np.arange(10, 20)])
    ).all()

    entry_slice = tss[:-20]
    assert entry_slice["start"] == pd.Period("2021-02-03", "D")
    assert (entry_slice["target"] == np.arange(80)).all()
    assert (
        entry_slice["feat_dynamic_real"] == np.array([np.arange(80)])
    ).all()

    entry_slice = tss[-20:]
    assert entry_slice["start"] == pd.Period("2021-02-03", "D") + 80
    assert (entry_slice["target"] == np.arange(80, 100)).all()
    assert (
        entry_slice["feat_dynamic_real"] == np.array([np.arange(80, 100)])
    ).all()


def test_split_mult_freq():
    splitter = DateSplitter(
        date=pd.Period("2021-01-01", "2h"),
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

    splitter = OffsetSplitter(offset=-7).split(dataset)

    assert [len(t["target"]) for t in splitter[0]] == [93, 43]
    assert [
        len(t["target"]) + len(s["target"])
        for t, s in splitter[1].generate_instances(prediction_length=7)
    ] == [100, 50]

    rolling_splitter = OffsetSplitter(offset=-21).split(dataset)

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


def check_training_validation(
    original_entry,
    train_entry,
    valid_pair,
    prediction_length: int,
    max_history,
    date,
    offset,
) -> None:
    assert original_entry[FieldName.ITEM_ID] == train_entry[FieldName.ITEM_ID]
    assert train_entry[FieldName.ITEM_ID] == valid_pair[0][FieldName.ITEM_ID]
    if max_history is not None:
        assert valid_pair[0][FieldName.TARGET].shape[-1] == max_history
    assert valid_pair[1][FieldName.TARGET].shape[-1] == prediction_length
    train_end = (
        train_entry[FieldName.START]
        + train_entry[FieldName.TARGET].shape[-1]
        * train_entry[FieldName.START].freq
    )
    if date is not None:
        assert train_end == date + train_entry[FieldName.START].freq
    if offset is not None:
        if offset > 0:
            assert train_entry[FieldName.TARGET].shape[-1] == offset
        else:
            assert (
                train_entry[FieldName.TARGET].shape[-1] - offset
                == original_entry[FieldName.TARGET].shape[-1]
            )
    assert train_end <= valid_pair[1][FieldName.START]
    valid_end = (
        valid_pair[0][FieldName.START]
        + valid_pair[0][FieldName.TARGET].shape[-1]
        * valid_pair[0][FieldName.START].freq
    )
    assert valid_end == valid_pair[1][FieldName.START]
    if FieldName.FEAT_DYNAMIC_REAL in valid_pair[0]:
        assert (
            valid_pair[0][FieldName.FEAT_DYNAMIC_REAL].shape[-1]
            == valid_pair[0][FieldName.TARGET].shape[-1] + prediction_length
        )


@pytest.mark.parametrize(
    "dataset",
    [
        ListDataset(
            [
                {
                    "item_id": 0,
                    "start": "2021-03-04",
                    "target": [1.0] * 365,
                    "feat_dynamic_real": [[2.0] * 365],
                },
                {
                    "item_id": 1,
                    "start": "2021-03-04",
                    "target": [2.0] * 265,
                    "feat_dynamic_real": [[3.0] * 265],
                },
            ],
            freq="D",
        ),
        ListDataset(
            [
                {
                    "item_id": 0,
                    "start": "2021-03-04",
                    "target": [[1.0] * 365, [10.0] * 365],
                    "feat_dynamic_real": [[2.0] * 365],
                },
                {
                    "item_id": 1,
                    "start": "2021-03-04",
                    "target": [[2.0] * 265, [20.0] * 265],
                    "feat_dynamic_real": [[3.0] * 265],
                },
            ],
            one_dim_target=False,
            freq="D",
        ),
    ],
)
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
def test_split(dataset, date, offset, windows, distance, max_history):
    assert (offset is None) != (date is None)

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
