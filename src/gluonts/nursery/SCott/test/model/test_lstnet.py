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
import pandas as pd

# First-party imports
from pts.dataset.artificial import constant_dataset
from pts.dataset import TrainDatasets, MultivariateGrouper
from pts.evaluation import backtest_metrics
from pts.model.lstnet import LSTNetEstimator
from pts import Trainer
from pts.evaluation import MultivariateEvaluator, make_evaluation_predictions


NUM_SERIES = 10
NUM_SAMPLES = 5


def load_multivariate_constant_dataset():
    metadata, train_ds, test_ds = constant_dataset()
    grouper_train = MultivariateGrouper(max_target_dim=NUM_SERIES)
    grouper_test = MultivariateGrouper(max_target_dim=NUM_SERIES)
    return TrainDatasets(
        metadata=metadata, train=grouper_train(train_ds), test=grouper_test(test_ds),
    )


dataset = load_multivariate_constant_dataset()
freq = dataset.metadata.metadata.freq
prediction_length = dataset.metadata.prediction_length


@pytest.mark.parametrize("skip_size", [1, 2])
@pytest.mark.parametrize("ar_window", [1, 2])
@pytest.mark.parametrize(
    "horizon, prediction_length",
    [[prediction_length, None], [None, prediction_length]],
)
def test_lstnet(skip_size, ar_window, horizon, prediction_length):
    estimator = LSTNetEstimator(
        skip_size=skip_size,
        ar_window=ar_window,
        num_series=NUM_SERIES,
        channels=6,
        kernel_size=2,
        context_length=4,
        freq=freq,
        horizon=horizon,
        prediction_length=prediction_length,
        trainer=Trainer(epochs=1, batch_size=2, learning_rate=0.01,),
    )

    predictor = estimator.train(dataset.train)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test, predictor=predictor, num_samples=NUM_SAMPLES
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    assert len(forecasts) == len(tss) == len(dataset.test)
    test_ds = dataset.test.list_data[0]
    for fct in forecasts:
        assert fct.freq == freq
        if estimator.horizon:
            assert fct.samples.shape == (NUM_SAMPLES, 1, NUM_SERIES)
        else:
            assert fct.samples.shape == (NUM_SAMPLES, prediction_length, NUM_SERIES,)
        assert (
            fct.start_date
            == pd.date_range(
                start=str(test_ds["start"]),
                periods=test_ds["target"].shape[1],  # number of test periods
                freq=freq,
                closed="right",
            )[-(horizon or prediction_length)]
        )

    evaluator = MultivariateEvaluator(
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    agg_metrics, item_metrics = evaluator(
        iter(tss), iter(forecasts), num_series=len(dataset.test)
    )
    assert agg_metrics["ND"] < 0.21
