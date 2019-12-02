# Standard library imports
import numpy as np

# First-party imports
import pytest

from gluonts.dataset.artificial import constant_dataset
from gluonts.distribution import (
    MultivariateGaussianOutput,
    LowrankMultivariateGaussianOutput,
)
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.dataset.common import TrainDatasets
from gluonts.trainer import Trainer
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.evaluation import MultivariateEvaluator


def load_multivariate_constant_dataset():
    dataset_info, train_ds, test_ds = constant_dataset()
    grouper_train = MultivariateGrouper(max_target_dim=10)
    grouper_test = MultivariateGrouper(num_test_dates=1, max_target_dim=10)
    metadata = dataset_info.metadata
    metadata.prediction_length = dataset_info.prediction_length
    return TrainDatasets(
        metadata=dataset_info.metadata,
        train=grouper_train(train_ds),
        test=grouper_test(test_ds),
    )


dataset = load_multivariate_constant_dataset()
target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)
metadata = dataset.metadata
estimator = DeepVAREstimator


@pytest.mark.timeout(20000)
@pytest.mark.parametrize(
    "distr_output, num_batches_per_epoch, Estimator, hybridize, use_copula",
    [
        (
            LowrankMultivariateGaussianOutput(dim=target_dim, rank=2),
            10,
            estimator,
            True,
            True,
        ),
        (
            LowrankMultivariateGaussianOutput(dim=target_dim, rank=2),
            10,
            estimator,
            False,
            False,
        ),
        (
            LowrankMultivariateGaussianOutput(dim=target_dim, rank=2),
            10,
            estimator,
            True,
            False,
        ),
        # fails with nan for now
        (
            MultivariateGaussianOutput(dim=target_dim),
            10,
            estimator,
            False,
            True,
        ),
        (
            MultivariateGaussianOutput(dim=target_dim),
            10,
            estimator,
            True,
            True,
        ),
    ],
)
def test_deepvar(
    distr_output, num_batches_per_epoch, Estimator, hybridize, use_copula
):

    estimator = Estimator(
        num_cells=20,
        num_layers=1,
        pick_incomplete=True,
        target_dim=target_dim,
        prediction_length=metadata.prediction_length,
        # target_dim=target_dim,
        freq=metadata.freq,
        distr_output=distr_output,
        scaling=False,
        use_copula=use_copula,
        trainer=Trainer(
            epochs=2,
            batch_size=8,
            learning_rate=1e-10,
            num_batches_per_epoch=num_batches_per_epoch,
            hybridize=hybridize,
        ),
    )

    agg_metrics, _ = backtest_metrics(
        train_dataset=dataset.train,
        test_dataset=dataset.test,
        forecaster=estimator,
        evaluator=MultivariateEvaluator(
            quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        ),
    )

    assert agg_metrics["ND"] < 1.5
