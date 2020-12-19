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

from mxnet import nd
import pytest

from gluonts.mx.block.quantile_output import QuantileLoss


@pytest.mark.parametrize(
    "quantile_weights, correct_qt_loss",
    [
        (None, [1.0, 1.8]),
        ([0.5, 0.5], 1.4),
    ],
)
def test_compute_quantile_loss(quantile_weights, correct_qt_loss) -> None:
    y_true = nd.ones(shape=(10, 10, 10))
    y_pred = nd.zeros(shape=(10, 10, 10, 2))

    quantiles = [0.5, 0.9]

    loss = QuantileLoss(quantiles, quantile_weights)
    tol = 1e-5
    if not quantile_weights:
        for idx, q in enumerate(quantiles):
            assert (
                nd.mean(
                    loss.compute_quantile_loss(
                        nd.ndarray, y_true, y_pred[:, :, :, idx], q
                    )
                )
                - correct_qt_loss[idx]
                < tol
            ), f"computing quantile loss at quantile {q} fails!"
    else:
        assert (
            nd.mean(loss(y_true, y_pred)) - correct_qt_loss < tol
        ), f"computing weighted quantile loss fails!"
