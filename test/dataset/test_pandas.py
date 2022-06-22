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

from gluonts.dataset import pandas


@pytest.fixture()
def my_series():
    idx = pd.date_range("2021-01-01", freq="1D", periods=3)
    series = pd.Series(np.random.normal(size=3), index=idx)
    return series


@pytest.fixture()
def my_dataframe(my_series):
    return my_series.to_frame(name="target")


# elements will be called with my_series fixture, doesn't include long-format
all_formats = [
    lambda series: series,
    lambda series: series.to_frame("target"),
    lambda series: 3 * [series],
    lambda series: 3 * [series.to_frame("target")],
    lambda series: {i: series for i in ["A", "B", "C"]},
    lambda series: {i: series.to_frame("target") for i in ["A", "B", "C"]},
]


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
    return pandas.PandasDataset.from_long_dataframe(
        dataframe=long_dataframe,
        target="target",
        timestamp="time",
        item_id="item",
        freq="1H",
        feat_dynamic_real=["dyn_real_1"],
        feat_static_cat=["stat_cat_1"],
    )


@pytest.mark.parametrize("get_data", all_formats)
def test_PandasDataset_init_with_all_formats(get_data, my_series):
    dataset = pandas.PandasDataset(dataframes=get_data(my_series))
    assert len(dataset)
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
    dataentry = pandas.as_dataentry(
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
        pandas.prepare_prediction_data(
            {"target": np.arange(20)}, ignore_last_n_targets=5
        )["target"]
        == np.arange(15)
    )


def test_prepare_prediction_data_nested():
    assert np.all(
        pandas.prepare_prediction_data(
            {"target": np.ones(shape=(3, 20))},
            ignore_last_n_targets=5,
        )["target"]
        == np.ones(shape=(3, 15))
    )


def test_prepare_prediction_data_with_features():
    res = pandas.prepare_prediction_data(
        {
            "start": pd.Period("2021-01-01", freq="1H"),
            "target": np.array([1.0, 2.0, np.nan]),
            "feat_dynamic_real": np.array([[1.0, 2.0, 3.0]]),
            "past_feat_dynamic_real": np.array([[1.0, 2.0, np.nan]]),
        },
        ignore_last_n_targets=1,
    )
    expected = {
        "start": pd.Period("2021-01-01", freq="1H"),
        "target": np.array([1.0, 2.0]),
        "feat_dynamic_real": np.array([[1.0, 2.0, 3.0]]),
        "past_feat_dynamic_real": np.array([[1.0, 2.0]]),
    }
    for key in res:
        assert np.all(res[key] == expected[key])


def test_check_timestamps():
    timestamps = ["2021-01-01 00:00", "2021-01-01 02:00", "2021-01-01 04:00"]
    assert pandas.check_timestamps(timestamps, freq="2H")


@pytest.mark.parametrize(
    "timestamps",
    [
        ["2021-01-01 00:00", "2021-01-01 02:00", "2021-01-01 02:00"],
        ["2021-01-01 00:00", "2021-01-01 02:00", "2021-01-01 03:00"],
        ["2021-01-01 04:00", "2021-01-01 02:00", "2021-01-01 00:00"],
    ],
)
def test_check_timestamps_fail(timestamps):
    assert not pandas.check_timestamps(timestamps, freq="2H")


def test_infer_timestamp(my_dataframe):
    ds = pandas.PandasDataset(my_dataframe, target="target", freq="1D")
    assert str(next(iter(ds))["start"]) == "2021-01-01"


def test_infer_timestamp2(my_dataframe):
    dfs = {"A": my_dataframe, "B": my_dataframe}
    ds = pandas.PandasDataset(dfs, target="target", freq="1D")
    assert str(next(iter(ds))["start"]) == "2021-01-01"


def test_infer_freq(my_dataframe):
    ds = pandas.PandasDataset(my_dataframe, target="target")
    assert ds.freq == "D"


def test_infer_freq2(my_dataframe):
    dfs = {"A": my_dataframe, "B": my_dataframe}
    ds = pandas.PandasDataset(dfs, target="target")
    assert ds.freq == "D"


def test_is_series(my_series):
    assert pandas.is_series(my_series)
    assert pandas.is_series([my_series])
    assert pandas.is_series({"A": my_series})


def test_is_series_fail(my_dataframe):
    with pytest.raises(AssertionError):
        assert pandas.is_series(my_dataframe)
    with pytest.raises(AssertionError):
        assert pandas.is_series([my_dataframe])
    with pytest.raises(AssertionError):
        assert pandas.is_series({"A": my_dataframe})


def test_series_to_dataframe(my_series):
    assert isinstance(pandas.series_to_dataframe(my_series), pd.DataFrame)
    assert isinstance(pandas.series_to_dataframe([my_series])[0], pd.DataFrame)
    dict_df = pandas.series_to_dataframe({"A": my_series})
    assert list(dict_df.keys())[0] == "A"
    assert isinstance(list(dict_df.values())[0], pd.DataFrame)
