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

from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.torch.model.mqdnn.estimator import (
    MQCNNEstimator,
    MQRNNEstimator,
)


@pytest.mark.parametrize(
    "estimator_constructor",
    [
        lambda dataset: MQCNNEstimator(
            freq=dataset.metadata.freq,
            prediction_length=dataset.metadata.prediction_length,
            context_length=50,
            quantiles=[0.1, 0.5, 0.9],
            batch_size=4,
            num_batches_per_epoch=3,
            trainer_kwargs=dict(max_epochs=2),
        ),
        lambda dataset: MQRNNEstimator(
            freq=dataset.metadata.freq,
            prediction_length=dataset.metadata.prediction_length,
            context_length=50,
            quantiles=[0.1, 0.5, 0.9],
            batch_size=4,
            num_batches_per_epoch=3,
            trainer_kwargs=dict(max_epochs=2),
        ),
    ],
)
def test_estimator_constant_dataset(estimator_constructor):
    constant = get_dataset("constant")

    estimator = estimator_constructor(constant)

    predictor = estimator.train(
        training_data=constant.train,
        validation_data=constant.train,
        shuffle_buffer_length=5,
    )

    forecasts = predictor.predict(constant.test)

    for forecast in forecasts:
        assert len(forecast.index) == predictor.prediction_length


@pytest.mark.parametrize(
    "estimator_constructor",
    [
        lambda freq, pred_len, num_fsr, num_fdr: MQCNNEstimator(
            freq=freq,
            prediction_length=pred_len,
            context_length=50,
            quantiles=[0.1, 0.5, 0.9],
            num_feat_static_real=num_fsr,
            num_feat_dynamic_real=num_fdr,
            batch_size=4,
            num_batches_per_epoch=3,
            trainer_kwargs=dict(max_epochs=2),
        ),
        lambda freq, pred_len, num_fsr, num_fdr: MQRNNEstimator(
            freq=freq,
            prediction_length=pred_len,
            context_length=50,
            quantiles=[0.1, 0.5, 0.9],
            num_feat_static_real=num_fsr,
            num_feat_dynamic_real=num_fdr,
            batch_size=4,
            num_batches_per_epoch=3,
            trainer_kwargs=dict(max_epochs=2),
        ),
    ],
)
def test_estimator_with_features(estimator_constructor):
    freq = "1h"
    prediction_length = 12
    num_feat_static_real = 1
    num_feat_dynamic_real = 3

    training_dataset = ListDataset(
        [
            {
                "start": "2021-01-01 00:00:00",
                "target": [1.0] * 200,
                "feat_static_cat": [0, 1],
                "feat_static_real": [42.0],
                "feat_dynamic_real": [[1.0] * 200] * 3,
            },
            {
                "start": "2021-02-01 00:00:00",
                "target": [1.0] * 100,
                "feat_static_cat": [1, 0],
                "feat_static_real": [1.0],
                "feat_dynamic_real": [[1.0] * 100] * 3,
            },
        ],
        freq=freq,
    )

    prediction_dataset = ListDataset(
        [
            {
                "start": "2021-01-01 00:00:00",
                "target": [1.0] * 200,
                "feat_static_cat": [0, 1],
                "feat_static_real": [42.0],
                "feat_dynamic_real": [[1.0] * (200 + prediction_length)] * 3,
            },
            {
                "start": "2021-02-01 00:00:00",
                "target": [1.0] * 100,
                "feat_static_cat": [1, 0],
                "feat_static_real": [1.0],
                "feat_dynamic_real": [[1.0] * (100 + prediction_length)] * 3,
            },
        ],
        freq=freq,
    )

    estimator = estimator_constructor(
        freq, prediction_length, num_feat_static_real, num_feat_dynamic_real
    )

    predictor = estimator.train(
        training_data=training_dataset,
        validation_data=training_dataset,
        shuffle_buffer_length=5,
    )

    forecasts = predictor.predict(prediction_dataset)

    for forecast in forecasts:
        assert len(forecast.index) == predictor.prediction_length
