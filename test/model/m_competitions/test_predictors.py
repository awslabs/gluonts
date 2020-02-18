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

# Standard library imports
import tempfile
from pathlib import Path

# Third-party imports
import pytest
from flaky import flaky

# First-party imports
from gluonts.dataset.artificial import constant_dataset
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.evaluation import Evaluator
from gluonts.model.predictor import Predictor
from gluonts.model.m_competitions import (
    Naive2Predictor,
    SeasonalNaivePredictor,
)


dataset_info, train_ds, test_ds = constant_dataset()
freq = dataset_info.metadata.freq
prediction_length = dataset_info.prediction_length


def seasonal_naive_predictor():
    return SeasonalNaivePredictor, dict(prediction_length=prediction_length)


def naive_02_predictor():
    return Naive2Predictor, dict(prediction_length=prediction_length)


@flaky(max_runs=3, min_passes=1)
@pytest.mark.parametrize(
    "RepresentablePredictor, parameters, accuracy",
    [seasonal_naive_predictor() + (0.0,), naive_02_predictor() + (0.0,)],
)
def test_accuracy(RepresentablePredictor, parameters, accuracy):
    estimator = RepresentablePredictor(freq=freq, **parameters)
    agg_metrics, item_metrics = backtest_metrics(
        train_dataset=train_ds,
        test_dataset=test_ds,
        forecaster=estimator,
        evaluator=Evaluator(calculate_owa=True),
    )

    assert agg_metrics["ND"] <= accuracy


@pytest.mark.parametrize(
    "RepresentablePredictor, parameters",
    [seasonal_naive_predictor(), naive_02_predictor()],
)
def test_seriali_predictors(RepresentablePredictor, parameters):
    predictor = RepresentablePredictor(freq=freq, **parameters)
    with tempfile.TemporaryDirectory() as temp_dir:
        predictor.serialize(Path(temp_dir))
        predictor_exp = Predictor.deserialize(Path(temp_dir))
        assert predictor == predictor_exp
