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

import pandas as pd

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.split import DateSplitter, OffsetSplitter
from gluonts.dataset.split.splitter import TimeSeriesSlice


@pytest.fixture(scope="session")
def dataset():
    return get_dataset("m4_hourly")


def test_date_splitter(dataset):
    prediction_length = dataset.metadata.prediction_length

    splitter = DateSplitter(
        prediction_length=prediction_length,
        split_date=pd.Timestamp("1750-01-05 04:00:00", freq="h"),
    )
    split = splitter.split(dataset.train)
    assert len(split.train[0][FieldName.TARGET]) + prediction_length == len(
        split.test[0][FieldName.TARGET]
    )


def test_offset_splitter(dataset):
    prediction_length = dataset.metadata.prediction_length

    max_history = 2 * prediction_length
    splitter = OffsetSplitter(
        prediction_length=prediction_length,
        split_offset=4 * prediction_length,
        max_history=max_history,
    )
    split = splitter.split(dataset.train)
    assert len(split.test[0][FieldName.TARGET]) == max_history
    assert len(split.train[0][FieldName.TARGET]) == 4 * prediction_length

    split = splitter.rolling_split(dataset.train, windows=3)
    for i in range(3):
        assert len(split.test[i][FieldName.TARGET]) == max_history
        assert len(split.train[i][FieldName.TARGET]) == 4 * prediction_length


def test_date_splitter_max_history(dataset):
    prediction_length = dataset.metadata.prediction_length

    max_history = 2 * prediction_length
    splitter = DateSplitter(
        prediction_length=prediction_length,
        split_date=pd.Timestamp("1750-01-05 04:00:00", freq="h"),
        max_history=max_history,
    )

    split = splitter.split(dataset.train)
    assert len(split.test[0][FieldName.TARGET]) == max_history

    split = splitter.rolling_split(dataset.train, windows=3)
    for i in range(3):
        assert len(split.test[i][FieldName.TARGET]) == max_history
