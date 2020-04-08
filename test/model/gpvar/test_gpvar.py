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

# Third-party imports
import pytest
from flaky import flaky
import mxnet as mx

# First-party imports
from gluonts.dataset.artificial import constant_dataset
from gluonts.distribution import LowrankMultivariateGaussian
from gluonts.distribution.lowrank_gp import LowrankGPOutput, GPArgProj
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.evaluation import MultivariateEvaluator
from gluonts.model.gpvar import GPVAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.common import TrainDatasets


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


def test_gp_output():
    # test that gp output gives expected shapes
    batch = 1
    hidden_size = 3
    dim = 4
    rank = 2

    states = mx.ndarray.ones(shape=(batch, dim, hidden_size))

    lowrank_gp_output = LowrankGPOutput(dim=dim, rank=rank)

    proj = lowrank_gp_output.get_args_proj()

    proj.initialize()

    distr_args = proj(states)

    mu, D, W = distr_args

    assert mu.shape == (batch, dim)
    assert D.shape == (batch, dim)
    assert W.shape == (batch, dim, rank)


def test_gpvar_proj():
    # test that gp proj gives expected shapes
    batch = 1
    hidden_size = 3
    dim = 4
    rank = 2

    states = mx.ndarray.ones(shape=(batch, dim, hidden_size))

    gp_proj = GPArgProj(rank=rank)
    gp_proj.initialize()

    distr_args = gp_proj(states)

    mu, D, W = distr_args

    assert mu.shape == (batch, dim)
    assert D.shape == (batch, dim)
    assert W.shape == (batch, dim, rank)

    distr = LowrankMultivariateGaussian(dim, rank, *distr_args)

    assert distr.mean.shape == (batch, dim)


@flaky(max_runs=3, min_passes=1)
@pytest.mark.parametrize("hybridize", [True, False])
@pytest.mark.parametrize("target_dim_sample", [None, 2])
@pytest.mark.parametrize("use_marginal_transformation", [True, False])
def test_smoke(
    hybridize: bool, target_dim_sample: int, use_marginal_transformation: bool
):
    num_batches_per_epoch = 1
    estimator = GPVAREstimator(
        distr_output=LowrankGPOutput(rank=2),
        num_cells=1,
        num_layers=1,
        pick_incomplete=True,
        prediction_length=metadata.prediction_length,
        target_dim=target_dim,
        target_dim_sample=target_dim_sample,
        freq=metadata.freq,
        use_marginal_transformation=use_marginal_transformation,
        trainer=Trainer(
            epochs=2,
            batch_size=10,
            learning_rate=1e-4,
            num_batches_per_epoch=num_batches_per_epoch,
            hybridize=hybridize,
        ),
    )

    agg_metrics, _ = backtest_metrics(
        train_dataset=dataset.train,
        test_dataset=dataset.test,
        forecaster=estimator,
        num_samples=10,
        evaluator=MultivariateEvaluator(
            quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        ),
    )
    assert agg_metrics["ND"] < 2.5
