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

from gluonts.dataset.artificial import constant_dataset
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.evaluation import backtest_metrics, MultivariateEvaluator
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.mx.distribution import (
    LowrankMultivariateGaussianOutput,
    MultivariateGaussianOutput,
)
from gluonts.mx.trainer import Trainer


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


@pytest.mark.parametrize(
    "distr_output, num_batches_per_epoch, Estimator, hybridize, "
    "use_marginal_transformation",
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
        (None, 10, estimator, True, True),
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
    distr_output,
    num_batches_per_epoch,
    Estimator,
    hybridize,
    use_marginal_transformation,
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
        use_marginal_transformation=use_marginal_transformation,
        trainer=Trainer(
            epochs=1,
            batch_size=8,
            learning_rate=1e-10,
            num_batches_per_epoch=num_batches_per_epoch,
            hybridize=hybridize,
        ),
    )

    predictor = estimator.train(training_data=dataset.train)

    agg_metrics, _ = backtest_metrics(
        test_dataset=dataset.test,
        predictor=predictor,
        evaluator=MultivariateEvaluator(
            quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        ),
    )

    assert agg_metrics["ND"] < 1.5
