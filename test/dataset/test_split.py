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

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import (
    OffsetSplitter,
    periods_between,
    split,
    slice_data_entry,
)


def test_time_series_slice():
    entry = {
        "start": pd.Period("2021-02-03", "D"),
        "target": np.array(range(100), dtype=float),
        "feat_dynamic_real": np.expand_dims(
            np.array(range(100), dtype=float), 0
        ),
    }

    entry_slice = slice_data_entry(entry, slice(10, 20))
    assert entry_slice["start"] == pd.Period("2021-02-03", "D") + 10
    assert (entry_slice["target"] == np.arange(10, 20)).all()
    assert (
        entry_slice["feat_dynamic_real"] == np.array([np.arange(10, 20)])
    ).all()

    entry_slice = slice_data_entry(entry, slice(None, -20))
    assert entry_slice["start"] == pd.Period("2021-02-03", "D")
    assert (entry_slice["target"] == np.arange(80)).all()
    assert (
        entry_slice["feat_dynamic_real"] == np.array([np.arange(80)])
    ).all()

    entry_slice = slice_data_entry(entry, slice(-20, None))
    assert entry_slice["start"] == pd.Period("2021-02-03", "D") + 80
    assert (entry_slice["target"] == np.arange(80, 100)).all()
    assert (
        entry_slice["feat_dynamic_real"] == np.array([np.arange(80, 100)])
    ).all()


@pytest.mark.parametrize(
    "start, end, count",
    [
        (
            pd.Period("2021-03-04", freq="2D"),
            pd.Period("2021-03-05", freq="2D"),
            1,
        ),
        (
            pd.Period("2021-03-04", freq="2D"),
            pd.Period("2021-03-08", freq="2D"),
            3,
        ),
        (
            pd.Period("2021-03-03 23:00", freq="30T"),
            pd.Period("2021-03-04 03:29", freq="30T"),
            9,
        ),
        (
            pd.Period("2015-04-07 00:00", freq="30T"),
            pd.Period("2015-04-07 09:31", "30T"),
            20,
        ),
        (
            pd.Period("2015-04-07 00:00", freq="30T"),
            pd.Period("2015-04-08 16:10", freq="30T"),
            81,
        ),
        (
            pd.Period("2021-01-01 00", freq="2H"),
            pd.Period("2021-01-01 08", "2H"),
            5,
        ),
        (
            pd.Period("2021-01-01 00", freq="2H"),
            pd.Period("2021-01-01 11", "2H"),
            6,
        ),
        (
            pd.Period("2021-03-04", freq="2D"),
            pd.Period("2021-03-02", freq="2D"),
            0,
        ),
        (
            pd.Period("2021-03-04", freq="2D"),
            pd.Period("2021-03-04", freq="2D"),
            1,
        ),
        (
            pd.Period("2021-03-03 23:00", freq="30T"),
            pd.Period("2021-03-03 03:29", freq="30T"),
            0,
        ),
    ],
)
def test_periods_between(start, end, count):
    assert count == periods_between(start, end)


def test_negative_offset_splitter():
    dataset = [
        {
            "item_id": 0,
            "start": pd.Period("2021-03-04", freq="D"),
            "target": np.ones(shape=(100,)),
        },
        {
            "item_id": 1,
            "start": pd.Period("2021-03-04", freq="D"),
            "target": 2 * np.ones(shape=(50,)),
        },
    ]

    train, test_gen = OffsetSplitter(offset=-7).split(dataset)

    assert [len(t["target"]) for t in train] == [93, 43]
    assert [
        len(t["target"]) + len(s["target"])
        for t, s in test_gen.generate_instances(prediction_length=7)
    ] == [100, 50]

    train, test_gen = OffsetSplitter(offset=-21).split(dataset)

    assert [len(t["target"]) for t in train] == [79, 29]
    assert [
        len(t["target"]) + len(s["target"])
        for t, s in test_gen.generate_instances(prediction_length=7, windows=3)
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
        [
            {
                "item_id": 0,
                "start": pd.Period("2021-03-04", freq="D"),
                "target": np.ones(shape=(365,)),
                "feat_dynamic_real": 2 * np.ones(shape=(1, 365)),
            },
            {
                "item_id": 1,
                "start": pd.Period("2021-03-04", freq="D"),
                "target": 2 * np.ones(shape=(265,)),
                "feat_dynamic_real": 3 * np.ones(shape=(1, 265)),
            },
        ],
        [
            {
                "item_id": 0,
                "start": pd.Period("2021-03-04", freq="D"),
                "target": np.stack(
                    [np.ones(shape=(365,)), 10 * np.ones(shape=(365,))]
                ),
                "feat_dynamic_real": 2 * np.ones(shape=(1, 365)),
            },
            {
                "item_id": 1,
                "start": pd.Period("2021-03-04", freq="D"),
                "target": np.stack(
                    [2 * np.ones(shape=(265,)), 20 * np.ones(shape=(265,))]
                ),
                "feat_dynamic_real": 3 * np.ones(shape=(1, 265)),
            },
        ],
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


@pytest.mark.parametrize(
    "entry, offset, prediction_length, test_label_start",
    [
        (
            {
                "start": pd.Period("2015-04-07 00:00:00", freq="30T"),
                "target": np.random.randn(100),
            },
            20,
            6,
            pd.Period("2015-04-07 10:00:00", freq="30T"),
        ),
        (
            {
                "start": pd.Period("2015-04-07 00:00:00", freq="30T"),
                "target": np.random.randn(100),
            },
            -20,
            6,
            pd.Period("2015-04-08 16:00:00", freq="30T"),
        ),
    ],
)
def test_split_offset(
    entry,
    offset,
    prediction_length,
    test_label_start,
):
    training_dataset, test_template = split([entry], offset=offset)

    training_entry = next(iter(training_dataset))
    test_input, test_label = next(
        iter(
            test_template.generate_instances(
                prediction_length=prediction_length
            )
        )
    )

    if offset < 0:
        training_size = (len(entry["target"]) + offset,)
    else:
        training_size = (offset,)

    assert training_entry["start"] == entry["start"]
    assert training_entry["target"].shape == training_size

    assert test_input["start"] == entry["start"]
    assert test_input["target"].shape == training_size

    assert test_label["start"] == test_label_start
    assert test_label["target"].shape == (prediction_length,)


@pytest.mark.parametrize(
    "entry, date, prediction_length, training_size",
    [
        (
            {
                "start": pd.Period("2015-04-07 00:00:00", freq="30T"),
                "target": np.random.randn(100),
            },
            pd.Period("2015-04-07 09:30", "30T"),
            6,
            (20,),
        ),
        (
            {
                "start": pd.Period("2015-04-07 00:00:00", freq="30T"),
                "target": np.random.randn(100),
            },
            pd.Period("2015-04-08 16:00:00", freq="30T"),
            6,
            (81,),
        ),
        (
            {
                "start": pd.Period("2021-01-01 00", freq="2H"),
                "target": np.arange(10),
            },
            pd.Period("2021-01-01 08", "2h"),
            2,
            (5,),
        ),
        (
            {
                "start": pd.Period("2021-01-01 00", freq="2H"),
                "target": np.arange(10),
            },
            pd.Period("2021-01-01 11", "2h"),
            2,
            (6,),
        ),
    ],
)
def test_split_date(
    entry,
    date,
    prediction_length,
    training_size,
):
    training_dataset, test_template = split([entry], date=date)

    training_entry = next(iter(training_dataset))
    test_input, test_label = next(
        iter(
            test_template.generate_instances(
                prediction_length=prediction_length
            )
        )
    )

    assert training_entry["start"] == entry["start"]
    assert training_entry["target"].shape == training_size

    assert test_input["start"] == entry["start"]
    assert test_input["target"].shape == training_size

    assert test_label["start"] == test_input["start"] + len(
        test_input["target"]
    )
    assert test_label["target"].shape == (prediction_length,)


@pytest.mark.parametrize(
    "dataset",
    [
        [
            {
                "start": pd.Period("2021-03-01", freq="D"),
                "target": np.ones(shape=(28,)),
            }
        ],
    ],
)
@pytest.mark.parametrize(
    "date, offset, windows, distance",
    [
        (pd.Period("2021-03-22", freq="D"), None, 1, None),
        (pd.Period("2021-03-21", freq="D"), None, 2, 1),
        (pd.Period("2021-03-21", freq="D"), None, 2, 7),
        (None, 22, 1, None),
        (None, 21, 2, 1),
        (None, 21, 2, 7),
        (None, -6, 1, None),
        (None, -7, 2, 1),
        (None, -7, 2, 7),
    ],
)
def test_invalid_offset(dataset, date, offset, windows, distance):
    assert (offset is None) != (date is None)
    exp_msg = "Not enough data to generate some of the windows"
    prediction_length = 7

    _, test_template = split(dataset, date=date, offset=offset)
    with pytest.raises(AssertionError) as excinfo:
        list(
            test_template.generate_instances(
                prediction_length=prediction_length,
                windows=windows,
                distance=distance,
            )
        )
    assert exp_msg in str(excinfo.value)
