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
import numpy as np

# First-party imports
from gluonts.dataset.artificial import constant_dataset
from gluonts.dataset.common import TrainDatasets
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.model.lstnet import LSTNetEstimator
from gluonts.trainer import Trainer
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.evaluation import MultivariateEvaluator


def load_multivariate_constant_dataset():
    metadata, train_ds, test_ds = constant_dataset()
    grouper_train = MultivariateGrouper(max_target_dim=10)
    grouper_test = MultivariateGrouper(max_target_dim=10)
    return TrainDatasets(
        metadata=metadata,
        train=grouper_train(train_ds),
        test=grouper_test(test_ds),
    )


dataset = load_multivariate_constant_dataset()


@pytest.mark.parametrize("skip_size", [1, 2])
@pytest.mark.parametrize("ar_window", [1, 2])
@pytest.mark.parametrize("hybridize", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lstnet(skip_size, ar_window, hybridize, dtype):
    estimator = LSTNetEstimator(
        skip_size=skip_size,
        ar_window=ar_window,
        num_series=10,
        channels=6,
        kernel_size=3,
        context_length=4,
        freq="1H",
        prediction_length=dataset.metadata.prediction_length,
        trainer=Trainer(
            epochs=1, batch_size=2, learning_rate=0.01, hybridize=hybridize
        ),
        dtype=dtype,
    )

    predictor = estimator.train(dataset.train)
    forecasts = list(predictor.predict(dataset.test))
    assert len(forecasts) == len(dataset.test)

    agg_metrics, _ = backtest_metrics(
        train_dataset=dataset.train,
        test_dataset=dataset.test,
        forecaster=predictor,
        evaluator=MultivariateEvaluator(
            quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        ),
    )
    assert agg_metrics["ND"] < 1.5
