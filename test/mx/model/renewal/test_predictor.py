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

import numpy as np
import pytest

from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.mx.model.renewal import DeepRenewalProcessEstimator
from gluonts.mx.model.renewal._predictor import (
    DeepRenewalProcessSampleOutputTransform,
)
from gluonts.mx import Trainer


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            [[[[3, 1, 2, 3, 1, 1, 1], [3, 5, 4, 1, 1, 1, 1]]]],
            [[[0, 0, 3, 5, 0, 4, 0]]],
        ),
        (
            [[[[7, 1, 2, 3, 1, 1, 1], [3, 5, 4, 1, 1, 1, 1]]]],
            [[[0, 0, 0, 0, 0, 0, 3]]],
        ),
        (
            [[[[1, 9, 2, 3, 1, 1, 1], [14, 5, 4, 1, 1, 1, 1]]]],
            [[[14, 0, 0, 0, 0, 0, 0]]],
        ),
        (
            [[[[8, 1, 2, 3, 1, 1, 1], [3, 5, 4, 1, 1, 1, 1]]]],
            [[[0, 0, 0, 0, 0, 0, 0]]],
        ),
        (
            [
                [
                    [[3, 1, 2, 3, 1, 1, 1], [3, 5, 4, 1, 1, 1, 1]],
                    [[3, 1, 2, 3, 1, 1, 1], [3, 5, 4, 1, 1, 1, 1]],
                ]
            ],
            [[[0, 0, 3, 5, 0, 4, 0], [0, 0, 3, 5, 0, 4, 0]]],
        ),
        (
            [
                [[[3, 1, 2, 3, 1, 1, 1], [3, 5, 4, 1, 1, 1, 1]]],
                [[[3, 2, 1, 1, 1, 1, 1], [6, 7, 8, 9, 1, 1, 1]]],
            ],
            [[[0, 0, 3, 5, 0, 4, 0]], [[0, 0, 6, 0, 7, 8, 9]]],
        ),
    ],
)
def test_output_transform(input, expected):
    expected = np.array(expected)
    tf = DeepRenewalProcessSampleOutputTransform()
    out = tf({}, np.array(input))

    assert np.allclose(out, expected)
    assert out.shape == expected.shape


def test_predictor_smoke_test():
    train_ds = ListDataset(
        [
            {
                "target": [
                    100.0,
                    63.0,
                    83.0,
                    126.0,
                    115.0,
                    92.0,
                    57.0,
                    95.0,
                    94.0,
                    92.0,
                    142.0,
                    35.0,
                    116.0,
                    78.0,
                    64.0,
                    141.0,
                ],
                "start": "2018-01-07 00:00:00",
                "feat_static_cat": [0],
            }
        ],
        freq="1m",
    )

    test_ds = ListDataset(
        [
            {
                "target": [100.0, 63.0, 83.0, 126.0, 115.0, 92.0, 57.0, 95.0]
                + [0] * 15,
                "start": "2018-01-07 00:00:00",
                "feat_static_cat": [1],
            }
        ],
        freq="1m",
    )

    estimator = DeepRenewalProcessEstimator(
        prediction_length=5,
        context_length=12,
        num_layers=1,
        num_cells=20,
        dropout_rate=0.1,
        trainer=Trainer(epochs=10),
    )

    predictor = estimator.train(train_ds)

    try:
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds,
            predictor=predictor,
            num_samples=200,
        )
        assert (np.min(x.samples) >= 0 for x in list(forecast_it))
    except IndexError:
        pytest.fail("Negative indices in renewal process predictor")
