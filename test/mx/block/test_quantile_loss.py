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

from gluonts.mx.block.quantile_output import (
    crps_weights_pwl,
    uniform_weights,
    QuantileLoss,
)


@pytest.mark.parametrize(
    "levels, weights_fun, actuals, predictions, expected_loss_values",
    [
        (
            [0.1, 0.5, 0.7, 0.95],
            uniform_weights,
            [10.0, 20.0, 30.0],
            [
                [11.0, 12.0, 13.0, 14.0],
                [19.0, 19.5, 21.0, 22.0],
                [27.0, 28.0, 29.0, 29.5],
            ],
            [0.375, 0.09375, 0.309375],
        ),
        (
            [0.1, 0.5, 0.7, 0.95],
            crps_weights_pwl,
            [10.0, 20.0, 30.0],
            [
                [11.0, 12.0, 13.0, 14.0],
                [19.0, 19.5, 21.0, 22.0],
                [27.0, 28.0, 29.0, 29.5],
            ],
            [0.35375002, 0.08750001, 0.28843752],
        ),
    ],
)
def test_quantile_loss(
    levels: list,
    actuals: list,
    predictions: list,
    expected_loss_values: list,
    weights_fun,
):
    loss_function = QuantileLoss(levels, quantile_weights=weights_fun(levels))
    loss_values = loss_function(
        mx.nd.array(actuals), mx.nd.array(predictions)
    ).asnumpy()
    assert np.allclose(expected_loss_values, loss_values)
