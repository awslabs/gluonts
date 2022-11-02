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

from gluonts.evaluation import backtest_metrics, MultivariateEvaluator
from gluonts.mx.model.deepvar_hierarchical import DeepVARHierarchicalEstimator
from gluonts.mx.trainer import Trainer


@pytest.mark.parametrize(
    "likelihood_weight, CRPS_weight, sample_LH, coherent_train_samples, coherent_pred_samples, warmstart_epoch_frac",
    [
        # Hier-E2E
        (
            0.0,
            1.0,
            False,
            True,
            True,
            0.0,
        ),
        # Hier-E2E with warm-start
        (
            0.0,
            1.0,
            False,
            True,
            True,
            0.3,
        ),
        # Hier-E2E with sample-likelihood loss
        (
            0.0,
            1.0,
            True,
            True,
            True,
            0.0,
        ),
        # Hier-E2E with warmstart and sample-likelihood loss
        (
            0.0,
            1.0,
            True,
            True,
            True,
            0.3,
        ),
        # DeepVAR+ (diagonal covariance)
        (
            1.0,
            0.0,
            False,
            False,
            True,
            0.0,
        ),
        # DeepVAR (diagonal covariance)
        (
            1.0,
            0.0,
            False,
            False,
            False,
            0.0,
        ),
    ],
)
def test_deepvar_hierarchical(
    sine7,
    likelihood_weight,
    CRPS_weight,
    sample_LH,
    coherent_train_samples,
    coherent_pred_samples,
    warmstart_epoch_frac,
):
    train_datasets = sine7()
    prediction_length = 10

    estimator = DeepVARHierarchicalEstimator(
        freq=train_datasets.metadata.freq,
        prediction_length=prediction_length,
        target_dim=train_datasets.metadata.S.shape[0],
        S=train_datasets.metadata.S,
        likelihood_weight=likelihood_weight,
        CRPS_weight=CRPS_weight,
        sample_LH=sample_LH,
        coherent_train_samples=coherent_train_samples,
        coherent_pred_samples=coherent_pred_samples,
        warmstart_epoch_frac=warmstart_epoch_frac,
        trainer=Trainer(
            epochs=10,
            num_batches_per_epoch=1,
            hybridize=False,
        ),
        num_samples_for_loss=10,
    )

    predictor = estimator.train(training_data=train_datasets.train)

    agg_metrics, _ = backtest_metrics(
        test_dataset=train_datasets.test,
        predictor=predictor,
        evaluator=MultivariateEvaluator(
            quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        ),
    )

    assert agg_metrics["ND"] < 1.5
