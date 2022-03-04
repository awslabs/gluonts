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

import tempfile
from itertools import islice
from pathlib import Path

import pytest

from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.model.predictor import Predictor
from gluonts.torch.model.deep_npts import (
    DeepNPTSEstimator,
    DeepNPTSNetwork,
    DeepNPTSNetworkDiscrete,
    DeepNPTSNetworkSmooth,
)


@pytest.mark.parametrize(
    "network_type", [DeepNPTSNetworkDiscrete, DeepNPTSNetworkSmooth]
)
def test_torch_deep_npts(network_type: DeepNPTSNetwork):
    constant = get_dataset("constant")

    estimator = DeepNPTSEstimator(
        freq=constant.metadata.freq,
        prediction_length=constant.metadata.prediction_length,
        context_length=2 * constant.metadata.prediction_length,
        network_type=network_type,
    )

    predictor = estimator.train(
        train_dataset=constant.train,
        epochs=5,
        batch_size=4,
        num_batches_per_epoch=3,
    )

    with tempfile.TemporaryDirectory() as td:
        predictor.serialize(Path(td))
        predictor_copy = Predictor.deserialize(Path(td))

    forecasts = predictor_copy.predict(constant.test)

    for f in islice(forecasts, 5):
        f.mean


@pytest.mark.parametrize(
    "network_type", [DeepNPTSNetworkDiscrete, DeepNPTSNetworkSmooth]
)
def test_torch_deep_npts_with_features(network_type: DeepNPTSNetwork):
    freq = "1h"
    prediction_length = 12

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
                "target": [1.0] * 200,
                "feat_static_cat": [1, 0],
                "feat_static_real": [1.0],
                "feat_dynamic_real": [[1.0] * 200] * 3,
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
                "target": [1.0] * 200,
                "feat_static_cat": [1, 0],
                "feat_static_real": [1.0],
                "feat_dynamic_real": [[1.0] * (200 + prediction_length)] * 3,
            },
        ],
        freq=freq,
    )

    estimator = DeepNPTSEstimator(
        freq=freq,
        prediction_length=prediction_length,
        context_length=2 * prediction_length,
        network_type=network_type,
        use_feat_static_cat=True,
        cardinality=[2, 2],
        num_feat_dynamic_real=0,
    )

    predictor = estimator.train(
        train_dataset=training_dataset,
        epochs=5,
        batch_size=4,
        num_batches_per_epoch=3,
    )

    with tempfile.TemporaryDirectory() as td:
        predictor.serialize(Path(td))
        predictor_copy = Predictor.deserialize(Path(td))

    forecasts = predictor_copy.predict(prediction_dataset)

    for f in islice(forecasts, 5):
        f.mean
