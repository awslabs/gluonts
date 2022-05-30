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
import numpy as np
import pytest

from gluonts.dataset import dataframe


@pytest.fixture()
def long_dataframe():
    N, T = 2, 10
    df = pd.DataFrame(index=np.arange(N * T))
    df["time"] = (
        2 * pd.date_range("2021-01-01 00:00", freq="1H", periods=T).to_list()
    )
    df["target"] = np.random.normal(size=(N * T,))
    df["item"] = T * ["A"] + T * ["B"]
    df["stat_cat_1"] = T * [0] + T * [1]
    df["dyn_real_1"] = np.random.normal(size=(N * T,))
    return df


@pytest.fixture
def long_dataset(long_dataframe):  # initialized with dict
    return dataframe.DataFramesDataset.from_long_dataframe(
        dataframe=long_dataframe,
        target="target",
        timestamp="time",
        item_id="item",
        freq="1H",
        feat_dynamic_real=["dyn_real_1"],
        feat_static_cat=["stat_cat_1"],
    )


def test_DataFramesDataset_init_with_list(long_dataframe):
    dataframes = list(map(lambda x: x[1], long_dataframe.groupby("item")))
    dataset = dataframe.DataFramesDataset(
        dataframes=dataframes,
        target="target",
        timestamp="time",
        freq="1H",
        feat_dynamic_real=["dyn_real_1"],
        feat_static_cat=["stat_cat_1"],
    )
    for i in dataset:
        assert isinstance(i, dict)


def test_DataFramesDataset_init_with_single_dataframe(long_dataframe):
    df = long_dataframe.loc[long_dataframe.loc[:, "item"] == "A", :]
    dataset = dataframe.DataFramesDataset(
        dataframes=df,
        target="target",
        timestamp="time",
        freq="1H",
        feat_dynamic_real=["dyn_real_1"],
        feat_static_cat=["stat_cat_1"],
    )
    for i in dataset:
        assert isinstance(i, dict)


def test_LongDataFrameDataset_iter(long_dataset):
    for i in long_dataset:
        assert isinstance(i, dict)
        assert "start" in i
        assert "target" in i
        assert "feat_dynamic_real" in i
        assert "feat_static_cat" in i


def test_LongDataFrameDataset_len(long_dataset):
    assert len(long_dataset) == 2


def test_as_dataentry(long_dataframe):
    df = long_dataframe.groupby("item").get_group("A")
    dataentry = dataframe.as_dataentry(
        data=df,
        target="target",
        timestamp="time",
        feat_dynamic_real=["dyn_real_1"],
        feat_static_cat=["stat_cat_1"],
    )
    assert "start" in dataentry
    assert "target" in dataentry
    assert "feat_dynamic_real" in dataentry
    assert "feat_static_cat" in dataentry


def test_prepare_prediction_data():
    assert np.all(
        dataframe.prepare_prediction_data(
            {"target": np.arange(20)}, ignore_last_n_targets=5
        )["target"]
        == np.arange(15)
    )


def test_prepare_prediction_data_nested():
    assert np.all(
        dataframe.prepare_prediction_data(
            {"target": np.ones(shape=(3, 20))},
            ignore_last_n_targets=5,
        )["target"]
        == np.ones(shape=(3, 15))
    )


def test_prepare_prediction_data_with_features():
    res = dataframe.prepare_prediction_data(
        {
            "start": pd.Timestamp("2021-01-01", freq="1H"),
            "target": np.array([1.0, 2.0, np.nan]),
            "feat_dynamic_real": np.array([[1.0, 2.0, 3.0]]),
            "past_feat_dynamic_real": np.array([[1.0, 2.0, np.nan]]),
        },
        ignore_last_n_targets=1,
    )
    expected = {
        "start": pd.Timestamp("2021-01-01", freq="1H"),
        "target": np.array([1.0, 2.0]),
        "feat_dynamic_real": np.array([[1.0, 2.0, 3.0]]),
        "past_feat_dynamic_real": np.array([[1.0, 2.0]]),
    }
    for key in res:
        assert np.all(res[key] == expected[key])


def test_check_timestamps():
    timestamps = ["2021-01-01 00:00", "2021-01-01 02:00", "2021-01-01 04:00"]
    assert dataframe.check_timestamps(timestamps, freq="2H")


@pytest.mark.parametrize(
    "timestamps",
    [
        ["2021-01-01 00:00", "2021-01-01 02:00", "2021-01-01 02:00"],
        ["2021-01-01 00:00", "2021-01-01 02:00", "2021-01-01 03:00"],
        ["2021-01-01 04:00", "2021-01-01 02:00", "2021-01-01 00:00"],
    ],
)
def test_check_timestamps_fail(timestamps):
    assert not dataframe.check_timestamps(timestamps, freq="2H")
