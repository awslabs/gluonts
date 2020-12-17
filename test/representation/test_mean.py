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

import mxnet as mx
import numpy as np
import pytest

from gluonts.mx.representation import MeanScaling

mean_cases = [
    (
        MeanScaling(),
        mx.nd.array(
            [
                [1.0] * 50,
                [0.0] * 25 + [3.0] * 25,
                [2.0] * 49 + [1.5] * 1,
                [0.0] * 50,
                [1.0] * 50,
            ]
        ),
        mx.nd.array(
            [
                [1.0] * 50,
                [0.0] * 25 + [1.0] * 25,
                [0.0] * 49 + [1.0] * 1,
                [1.0] * 50,
                [0.0] * 50,
            ]
        ),
        mx.nd.array([1.0, 3.0, 1.5, 1.00396824, 1.00396824]),
    ),
    (
        MeanScaling(),
        mx.nd.array(
            [
                [120.0] * 25 + [150.0] * 25,
                [0.0] * 10 + [3.0] * 20 + [61.0] * 20,
                [0.0] * 50,
                [2e-2] * 10 + [0.0] * 30 + [3e-2] * 10,
            ]
        ),
        mx.nd.array(
            [
                [1.0] * 25 + [1.0] * 25,
                [0.0] * 10 + [1.0] * 20 + [1.0] * 20,
                [0.0] * 50,
                [1.0] * 10 + [0.0] * 30 + [1.0] * 10,
            ]
        ),
        mx.nd.array([135.0, 32.0, 73.00454712, 2.5e-2]),
    ),
    (
        MeanScaling(),
        mx.nd.random.normal(shape=(5, 30)),
        mx.nd.zeros(shape=(5, 30)),
        1e-10 * mx.nd.ones(shape=(5,)),
    ),
]


@pytest.mark.parametrize("s, target, observed, expected_scale", mean_cases)
def test_mean(s, target, observed, expected_scale):
    target_scaled, scale, _ = s(target, observed, None, [])
    scale = mx.nd.reshape(scale, shape=(len(scale),))

    assert np.allclose(
        expected_scale.asnumpy(), scale.asnumpy()
    ), "mismatch in the scale computation"

    expected_target_scaled = mx.nd.broadcast_div(
        target, expected_scale.expand_dims(axis=1)
    )

    assert np.allclose(
        expected_target_scaled.asnumpy(), target_scaled.asnumpy()
    ), "mismatch in the scaled target computation"
