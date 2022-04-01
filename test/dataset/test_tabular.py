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

from gluonts.model.deepar import DeepAREstimator
from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.dataset import tabular


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
def long_dataset(long_dataframe):
    return tabular.DataFrameDataset(
        data=long_dataframe,
        target="target",
        timestamp="time",
        item_id="item",
        freq="1H",
        feat_dynamic_real=["dyn_real_1"],
        feat_static_cat=["stat_cat_1"],
    )


def test_DataFrameDataset_iter(long_dataset):
    for i in long_dataset:
        assert isinstance(i, dict)
        assert "start" in i
        assert "target" in i


def test_DataFrameDataset_len(long_dataset):
    assert len(long_dataset) == 2


def test_DataFrameDataset_train_eval_loop(long_dataset):
    freq, pl = long_dataset.freq, 5
    model = DeepAREstimator(
        freq=freq,
        prediction_length=pl,
        batch_size=2,
        trainer=Trainer(epochs=1),
    )
    predictor = model.train(long_dataset)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=long_dataset,
        predictor=predictor,
        num_samples=2,
    )
    predictions = list(predictor.predict(long_dataset))
    evaluator = Evaluator(quantiles=[0.5])
    agg_metrics, item_metrics = evaluator(
        ts_it, forecast_it, num_series=len(long_dataset)
    )


def test_dataframeTS_to_dataentry(long_dataframe):
    df = long_dataframe.groupby("item").get_group("A")
    dataentry = tabular.dataframeTS_to_dataentry(
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
    assert tabular.prepare_prediction_data(
        {"target": np.arange(20).tolist()}, prediction_length=5
    ) == {"target": np.arange(15).tolist()}


def test_prepare_prediction_data_nested():
    assert tabular.prepare_prediction_data(
        {"target": 2 * [np.arange(20).tolist()]},
        prediction_length=5,
    ) == {"target": 2 * [np.arange(15).tolist()]}


def test_prepare_prediction_data_with_features():
    assert tabular.prepare_prediction_data(
        {
            "start": "2021-01-01",
            "target": [1.0, 2.0, np.nan],
            "feat_dynamic_real": [[1.0, 2.0, 3.0]],
            "past_feat_dynamic_real": [[1.0, 2.0, np.nan]],
        },
        prediction_length=1,
    ) == {
        "start": "2021-01-01",
        "target": [1.0, 2.0],
        "feat_dynamic_real": [[1.0, 2.0, 3.0]],
        "past_feat_dynamic_real": [[1.0, 2.0]],
    }


def test_check_timestamps():
    timestamps = ["2021-01-01 00:00", "2021-01-01 02:00", "2021-01-01 04:00"]
    tabular.check_timestamps(timestamps, freq="2H")


@pytest.mark.parametrize(
    "timestamps",
    [
        ["2021-01-01 00:00", "2021-01-01 02:00", "2021-01-01 02:00"],
        ["2021-01-01 00:00", "2021-01-01 02:00", "2021-01-01 03:00"],
        ["2021-01-01 04:00", "2021-01-01 02:00", "2021-01-01 00:00"],
    ],
)
def test_check_timestamps_fail(timestamps):
    with pytest.raises(AssertionError):
        tabular.check_timestamps(timestamps, freq="2H")
