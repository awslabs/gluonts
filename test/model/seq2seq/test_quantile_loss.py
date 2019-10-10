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
from mxnet import nd

# First-party imports
from gluonts.block.quantile_output import QuantileLoss


def test_compute_quantile_loss() -> None:
    y_true = nd.ones(shape=(10, 10, 10))
    y_pred = nd.zeros(shape=(10, 10, 10, 2))

    quantiles = [0.5, 0.9]

    loss = QuantileLoss(quantiles)

    correct_qt_loss = [1.0, 1.8]

    for idx, q in enumerate(quantiles):
        assert (
            nd.mean(
                loss.compute_quantile_loss(
                    nd.ndarray, y_true, y_pred[:, :, :, idx], q
                )
            )
            - correct_qt_loss[idx]
            < 1e-5
        ), f"computing quantile loss at quantile {q} fails!"
