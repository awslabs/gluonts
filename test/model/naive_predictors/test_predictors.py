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

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import pytest
from flaky import flaky
from pydantic import PositiveInt

from gluonts.dataset.artificial import constant_dataset
from gluonts.dataset.common import Dataset
from gluonts.evaluation import backtest_metrics, Evaluator
from gluonts.model.naive_2 import Naive2Predictor
from gluonts.model.predictor import Predictor
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.support.pandas import forecast_start


def generate_random_dataset(
    num_ts: int, start_time: str, freq: str, min_length: int, max_length: int
) -> Dataset:
    start_timestamp = pd.Timestamp(start_time, freq=freq)
    for _ in range(num_ts):
        ts_length = np.random.randint(low=min_length, high=max_length)
        target = np.random.uniform(size=(ts_length,))
        data = {"target": target, "start": start_timestamp}
        yield data


PREDICTION_LENGTH = PositiveInt(30)
SEASON_LENGTH = PositiveInt(210)
START_TIME = "2018-01-03 14:37:12"  # That's a Wednesday
MIN_LENGTH = 300
MAX_LENGTH = 400
NUM_TS = 10


@pytest.mark.parametrize(
    "predictor_cls", [SeasonalNaivePredictor, Naive2Predictor]
)
@pytest.mark.parametrize(
    "freq", ["1min", "15min", "30min", "1H", "2H", "12H", "7D", "1W", "1M"]
)
def test_predictor(predictor_cls, freq: str):
    predictor = predictor_cls(
        freq=freq,
        prediction_length=PREDICTION_LENGTH,
        season_length=SEASON_LENGTH,
    )
    dataset = list(
        generate_random_dataset(
            num_ts=NUM_TS,
            start_time=START_TIME,
            freq=freq,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH,
        )
    )

    # get forecasts
    forecasts = list(predictor.predict(dataset))

    assert len(dataset) == NUM_TS
    assert len(forecasts) == NUM_TS

    # check forecasts are as expected
    for data, forecast in zip(dataset, forecasts):
        assert forecast.samples.shape == (1, PREDICTION_LENGTH)

        ref = data["target"][
            -SEASON_LENGTH : -SEASON_LENGTH + PREDICTION_LENGTH
        ]

        assert forecast.start_date == forecast_start(data)

        # specifically for the seasonal naive we can test the supposed result directly
        if predictor_cls == SeasonalNaivePredictor:
            assert np.allclose(forecast.samples[0], ref)


# CONSTANT DATASET TESTS:


dataset_info, constant_train_ds, constant_test_ds = constant_dataset()
CONSTANT_DATASET_FREQ = dataset_info.metadata.freq
CONSTANT_DATASET_PREDICTION_LENGTH = dataset_info.prediction_length


def seasonal_naive_predictor():
    return (
        SeasonalNaivePredictor,
        dict(prediction_length=CONSTANT_DATASET_PREDICTION_LENGTH),
    )


def naive_2_predictor():
    return (
        Naive2Predictor,
        dict(prediction_length=CONSTANT_DATASET_PREDICTION_LENGTH),
    )


@flaky(max_runs=3, min_passes=1)
@pytest.mark.parametrize(
    "predictor_cls, parameters, accuracy",
    [seasonal_naive_predictor() + (0.0,), naive_2_predictor() + (0.0,)],
)
def test_accuracy(predictor_cls, parameters, accuracy):
    predictor = predictor_cls(freq=CONSTANT_DATASET_FREQ, **parameters)
    agg_metrics, item_metrics = backtest_metrics(
        test_dataset=constant_test_ds,
        predictor=predictor,
        evaluator=Evaluator(calculate_owa=True),
    )

    assert agg_metrics["ND"] <= accuracy


# SERIALIZATION/DESERIALIZATION TESTS:


@pytest.mark.parametrize(
    "predictor_cls, parameters",
    [seasonal_naive_predictor(), naive_2_predictor()],
)
def test_seriali_predictors(predictor_cls, parameters):
    predictor = predictor_cls(freq=CONSTANT_DATASET_FREQ, **parameters)
    with tempfile.TemporaryDirectory() as temp_dir:
        predictor.serialize(Path(temp_dir))
        predictor_exp = Predictor.deserialize(Path(temp_dir))
        assert predictor == predictor_exp
