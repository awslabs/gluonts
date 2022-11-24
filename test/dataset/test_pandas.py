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

import io
from typing import Callable, List, Union

import pandas as pd
import numpy as np
import pytest

from gluonts.dataset import pandas
from gluonts.testutil.equality import assert_recursively_equal


@pytest.fixture(params=[pd.date_range, pd.period_range])
def my_series(request):

    idx = request.param("2021-01-01", freq="1D", periods=3)
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


def test_is_uniform_2H():
    timestamps = ["2021-01-01 00:00", "2021-01-01 02:00", "2021-01-01 04:00"]
    assert pandas.is_uniform(pd.DatetimeIndex(timestamps).to_period("2H"))


@pytest.mark.parametrize(
    "timestamps",
    [
        ["2021-01-01 00:00", "2021-01-01 02:00", "2021-01-01 02:00"],
        ["2021-01-01 00:00", "2021-01-01 02:00", "2021-01-01 03:00"],
        ["2021-01-01 04:00", "2021-01-01 02:00", "2021-01-01 00:00"],
    ],
)
def test_is_uniform_2H_fail(timestamps):
    assert not pandas.is_uniform(pd.DatetimeIndex(timestamps).to_period("2H"))


def test_infer_period(my_dataframe):
    ds = pandas.PandasDataset(my_dataframe, target="target", freq="1D")
    for entry in ds:
        assert entry["start"] == pd.Period("2021-01-01", freq="1D")


def test_infer_period2(my_dataframe):
    dfs = {"A": my_dataframe, "B": my_dataframe}
    ds = pandas.PandasDataset(dfs, target="target", freq="1D")
    for entry in ds:
        assert entry["start"] == pd.Period("2021-01-01", freq="1D")


def test_long_csv_3M():
    data = (
        "timestamp,item_id,target\n"
        "2021-03,0,102\n"
        "2022-07,1,138\n"
        "2021-06,0,103\n"
        "2021-11,2,227\n"
        "2021-09,0,102\n"
        "2021-12,0,99\n"
        "2022-01,1,148\n"
        "2022-05,2,229\n"
        "2021-04,1,134\n"
        "2022-04,1,117\n"
        "2021-02,2,212\n"
        "2021-05,2,225\n"
        "2021-07,1,151\n"
        "2021-08,2,221\n"
        "2022-02,2,230\n"
        "2021-10,1,144\n"
    )

    expected_entries = [
        {
            "start": pd.Period("2021-03", freq="3M"),
            "item_id": 0,
            "target": np.array([102, 103, 102, 99]),
        },
        {
            "start": pd.Period("2021-04", freq="3M"),
            "item_id": 1,
            "target": np.array([134, 151, 144, 148, 117, 138]),
        },
        {
            "start": pd.Period("2021-02", freq="3M"),
            "item_id": 2,
            "target": np.array([212, 225, 221, 227, 230, 229]),
        },
    ]

    with io.StringIO(data) as fp:
        ds = pandas.PandasDataset.from_long_dataframe(
            pd.read_csv(fp),
            target="target",
            item_id="item_id",
            timestamp="timestamp",
            freq="3M",
        )
    for entry, expected_entry in zip(ds, expected_entries):
        assert entry["start"].freqstr == "3M"
        assert entry["start"] == expected_entry["start"]
        assert entry["item_id"] == expected_entry["item_id"]
        assert np.allclose(entry["target"], expected_entry["target"])

    with io.StringIO(data) as fp:
        ds = pandas.PandasDataset.from_long_dataframe(
            pd.read_csv(fp, index_col="timestamp"),
            target="target",
            item_id="item_id",
            freq="3M",
        )
    for entry, expected_entry in zip(ds, expected_entries):
        assert entry["start"].freqstr == "3M"
        assert entry["start"] == expected_entry["start"]
        assert entry["item_id"] == expected_entry["item_id"]
        assert np.allclose(entry["target"], expected_entry["target"])


def _testcase_series(freq: str, index_type: Callable, dtype=np.float32):
    series = [
        pd.Series(
            np.arange(10, dtype=dtype),
            index_type("2021-01-01 00:00:00", periods=10, freq=freq),
        ),
        pd.Series(
            np.arange(20, dtype=dtype),
            index_type("2021-01-02 00:00:00", periods=20, freq=freq),
        ),
        pd.Series(
            np.arange(30, dtype=dtype),
            index_type("2021-01-03 00:00:00", periods=30, freq=freq),
        ),
    ]

    dataset = pandas.PandasDataset(series)

    expected_entries = [
        {
            "start": pd.Period(s.index[0], freq=freq),
            "target": s.values,
        }
        for s in series
    ]

    return dataset, expected_entries


def _testcase_dataframes_without_index(
    freq: str,
    target: Union[str, List[str]],
    feat_dynamic_real: List[str],
    dtype=np.float32,
):
    dataframes = [
        pd.DataFrame.from_dict(
            {
                "timestamp": pd.period_range(
                    "2021-01-01 00:00:00", periods=10, freq=freq
                )
                .map(str)
                .to_list(),
                "A": 1 + np.arange(10, dtype=dtype),
                "B": 2 + np.arange(10, dtype=dtype),
                "C": 3 + np.arange(10, dtype=dtype),
            }
        ),
        pd.DataFrame.from_dict(
            {
                "timestamp": pd.period_range(
                    "2021-01-02 00:00:00", periods=20, freq=freq
                )
                .map(str)
                .to_list(),
                "A": 1 + np.arange(20, dtype=dtype),
                "B": 2 + np.arange(20, dtype=dtype),
                "C": 3 + np.arange(20, dtype=dtype),
            }
        ),
        pd.DataFrame.from_dict(
            {
                "timestamp": pd.period_range(
                    "2021-01-03 00:00:00", periods=30, freq=freq
                )
                .map(str)
                .to_list(),
                "A": 1 + np.arange(30, dtype=dtype),
                "B": 2 + np.arange(30, dtype=dtype),
                "C": 3 + np.arange(30, dtype=dtype),
            }
        ),
    ]

    dataset = pandas.PandasDataset(
        dataframes,
        timestamp="timestamp",
        freq=freq,
        target=target,
        feat_dynamic_real=feat_dynamic_real,
    )

    expected_entries = [
        {
            "start": pd.Period(df["timestamp"][0], freq=freq),
            "target": df[target].values.transpose(),
            "feat_dynamic_real": df[feat_dynamic_real].values.transpose(),
        }
        for df in dataframes
    ]

    return dataset, expected_entries


def _testcase_dataframes_with_index(
    freq: str,
    index_type: Callable,
    target: Union[str, List[str]],
    feat_dynamic_real: List[str],
    dtype=np.float32,
):
    dataframes = [
        pd.DataFrame.from_dict(
            {
                "timestamp": index_type(
                    "2021-01-01 00:00:00", periods=10, freq=freq
                ),
                "A": 1 + np.arange(10, dtype=dtype),
                "B": 2 + np.arange(10, dtype=dtype),
                "C": 3 + np.arange(10, dtype=dtype),
            }
        ).set_index("timestamp"),
        pd.DataFrame.from_dict(
            {
                "timestamp": index_type(
                    "2021-01-02 00:00:00", periods=20, freq=freq
                ),
                "A": 1 + np.arange(20, dtype=dtype),
                "B": 2 + np.arange(20, dtype=dtype),
                "C": 3 + np.arange(20, dtype=dtype),
            }
        ).set_index("timestamp"),
        pd.DataFrame.from_dict(
            {
                "timestamp": index_type(
                    "2021-01-03 00:00:00", periods=30, freq=freq
                ),
                "A": 1 + np.arange(30, dtype=dtype),
                "B": 2 + np.arange(30, dtype=dtype),
                "C": 3 + np.arange(30, dtype=dtype),
            }
        ).set_index("timestamp"),
    ]

    print(type(dataframes[0].index))

    dataset = pandas.PandasDataset(
        dataframes,
        target=target,
        feat_dynamic_real=feat_dynamic_real,
    )

    expected_entries = [
        {
            "start": pd.Period(df.index[0], freq=freq),
            "target": df[target].values.transpose(),
            "feat_dynamic_real": df[feat_dynamic_real].values.transpose(),
        }
        for df in dataframes
    ]

    return dataset, expected_entries


@pytest.mark.parametrize(
    "dataset, expected_entries",
    [
        _testcase_series(freq="D", index_type=pd.period_range),
        _testcase_series(freq="H", index_type=pd.date_range),
        _testcase_dataframes_without_index(
            freq="D",
            target="A",
            feat_dynamic_real=["B", "C"],
        ),
        _testcase_dataframes_without_index(
            freq="H",
            target=["A", "B"],
            feat_dynamic_real=["C"],
        ),
        _testcase_dataframes_with_index(
            freq="D",
            index_type=pd.period_range,
            target="A",
            feat_dynamic_real=["B", "C"],
        ),
        _testcase_dataframes_with_index(
            freq="H",
            index_type=pd.date_range,
            target=["A", "B"],
            feat_dynamic_real=["C"],
        ),
    ],
)
def test_pandas_dataset_cases(dataset, expected_entries):
    for entry, expected_entry in zip(dataset, expected_entries):
        assert_recursively_equal(entry, expected_entry)
