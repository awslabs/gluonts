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

import logging
import math
from pathlib import Path

import pytest

import gluonts
from gluonts.core.component import equals
from gluonts.core.serde import dump_code, load_code
from gluonts.dataset.artificial import constant_dataset
from gluonts.dataset.stat import ScaleHistogram  # noqa
from gluonts.dataset.stat import (
    DatasetStatistics,
    calculate_dataset_statistics,
)
from gluonts.evaluation import backtest_metrics, Evaluator
from gluonts.evaluation.backtest import BacktestInformation
from gluonts.model.trivial.mean import MeanEstimator

root = logging.getLogger()
root.setLevel(logging.DEBUG)


def make_estimator(freq, prediction_length):
    # noinspection PyTypeChecker
    return MeanEstimator(
        prediction_length=prediction_length, freq=freq, num_samples=5
    )


def test_forecast_parser():
    # verify that logged for estimator, datasets and metrics can be recovered
    # from their string representation

    dataset_info, train_ds, test_ds = constant_dataset()

    estimator = make_estimator(
        dataset_info.metadata.freq, dataset_info.prediction_length
    )
    assert repr(estimator) == repr(load_code(repr(estimator)))

    predictor = estimator.train(training_data=train_ds)

    stats = calculate_dataset_statistics(train_ds)
    assert stats == eval(
        repr(stats), globals(), {"gluonts": gluonts}
    )  # TODO: use load

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = backtest_metrics(test_ds, predictor, evaluator)

    # reset infinite metrics to 0 (otherwise the assertion below fails)
    for key, val in agg_metrics.items():
        if not math.isfinite(val):
            agg_metrics[key] = 0.0

    assert agg_metrics == load_code(dump_code(agg_metrics))


@pytest.mark.skip()
def test_benchmark(caplog):
    # makes sure that information logged can be reconstructed from previous
    # logs

    with caplog.at_level(logging.DEBUG):
        dataset_info, train_ds, test_ds = constant_dataset()

        estimator = make_estimator(
            dataset_info.metadata.freq, dataset_info.prediction_length
        )
        predictor = estimator.train(training_data=train_ds)
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        backtest_metrics(test_ds, predictor, evaluator)
        train_stats = calculate_dataset_statistics(train_ds)
        test_stats = calculate_dataset_statistics(test_ds)

    log_info = BacktestInformation.make_from_log_contents(caplog.text)

    assert train_stats == log_info.train_dataset_stats
    assert test_stats == log_info.test_dataset_stats
    assert equals(estimator, log_info.estimator)

    print(log_info)
