# Standard library imports
import logging
import math
from pathlib import Path

# First-party imports
import gluonts  # noqa
from gluonts.core.component import equals
from gluonts.core.serde import load_code, dump_code
from gluonts.dataset.artificial import constant_dataset
from gluonts.dataset.stat import (  # noqa
    DatasetStatistics,
    ScaleHistogram,
    calculate_dataset_statistics,
)
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import BacktestInformation, backtest_metrics
from gluonts.model.testutil import MeanEstimator

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

    stats = calculate_dataset_statistics(train_ds)
    assert stats == eval(repr(stats))  # TODO: use load

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = backtest_metrics(train_ds, test_ds, estimator, evaluator)

    # reset infinite metrics to 0 (otherwise the assertion below fails)
    for key, val in agg_metrics.items():
        if not math.isfinite(val):
            agg_metrics[key] = 0.0

    assert agg_metrics == load_code(dump_code(agg_metrics))


def test_benchmark(caplog):
    # makes sure that information logged can be reconstructed from previous
    # logs

    caplog.set_level(logging.DEBUG, logger='log.txt')

    dataset_info, train_ds, test_ds = constant_dataset()

    estimator = make_estimator(
        dataset_info.metadata.freq, dataset_info.prediction_length
    )
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    backtest_metrics(train_ds, test_ds, estimator, evaluator)
    train_stats = calculate_dataset_statistics(train_ds)
    test_stats = calculate_dataset_statistics(test_ds)
    log_file = str(Path(__file__).parent / 'log.txt')
    log_info = BacktestInformation.make_from_log(log_file)

    assert train_stats == log_info.train_dataset_stats
    assert test_stats == log_info.test_dataset_stats
    assert equals(estimator, log_info.estimator)

    print(log_info)
