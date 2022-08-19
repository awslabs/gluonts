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
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Optional

import pytest

from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import Predictor
from gluonts.torch.model.deep_npts import (
    DeepNPTSEstimator,
    DeepNPTSNetwork,
    DeepNPTSNetworkDiscrete,
    DeepNPTSNetworkSmooth,
)


@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize(
    "input_scaling", [None, "min_max_scaling", "standard_normal_scaling"]
)
@pytest.mark.parametrize("dropout_rate", [None, 0.0, 0.1])
@pytest.mark.parametrize(
    "network_type",
    [
        partial(DeepNPTSNetworkDiscrete, use_softmax=True),
        partial(DeepNPTSNetworkDiscrete, use_softmax=False),
        DeepNPTSNetworkSmooth,
    ],
)
def test_torch_deep_npts_with_features(
    batch_norm: bool,
    input_scaling: Optional[str],
    dropout_rate: Optional[float],
    network_type: DeepNPTSNetwork,
):
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
        batch_norm=batch_norm,
        network_type=network_type,
        use_feat_static_cat=True,
        cardinality=[2, 2],
        num_feat_static_real=1,
        num_feat_dynamic_real=0,
        input_scaling=input_scaling,
        dropout_rate=dropout_rate,
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
