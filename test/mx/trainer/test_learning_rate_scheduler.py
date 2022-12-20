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

from gluonts.mx.trainer import learning_rate_scheduler as lrs


@pytest.mark.parametrize(
    "base_lr, decay_factor, patience, minimum_lr, seq_loss_lr",
    [
        (
            1.0,
            0.5,
            3,  # decrease if no best score is found every 3 steps
            0.1,
            [
                (10, 1.0),
                (9, 1.0),
                (8, 1.0),
                (7, 1.0),
                (7, 1.0),
                (7, 1.0),
                (7, 0.5),
                (7, 0.5),
                (6, 0.5),
                (7, 0.5),
                (5, 0.5),
                (4, 0.5),
                (5, 0.5),
                (6, 0.5),
                (6, 0.25),
                (5, 0.25),
                (4, 0.25),
                (3, 0.25),
                (3, 0.25),
                (4, 0.25),
                (4, 0.125),
                (4, 0.125),
                (2, 0.125),
                (3, 0.125),
                (2, 0.125),
                (3, 0.1),
                (2, 0.1),
                (2, 0.1),
            ],
        ),
        (
            1e-2,
            0.5,
            1,  # decrease as soon as a not-best-score is observed
            1e-3,
            [
                (1.00, 0.01),
                (0.90, 0.01),
                (0.80, 0.01),
                (0.85, 0.005),
                (0.82, 0.0025),
                (0.75, 0.0025),
                (0.75, 0.00125),
                (0.60, 0.00125),
                (0.70, 0.001),
                (0.58, 0.001),
                (0.59, 0.001),
            ],
        ),
        (
            1e-2,
            0.9,
            0,  # decrease at every step regardless of the score
            2e-3,
            [
                (1.0, 0.009000000000000001),
                (1.1, 0.008100000000000001),
                (0.9, 0.007290000000000001),
                (0.8, 0.006561),
                (1.2, 0.005904900000000001),
                (1.0, 0.00531441),
                (0.9, 0.004782969000000001),
                (0.8, 0.004304672100000001),
                (0.4, 0.003874204890000001),
                (5.0, 0.003486784401000001),
                (0.5, 0.0031381059609000006),
                (1.0, 0.0028242953648100013),
                (2.0, 0.002541865828329001),
                (1.9, 0.002287679245496101),
                (1.8, 0.002058911320946491),
                (0.3, 0.002),
                (0.2, 0.002),
                (1.0, 0.002),
                (1.2, 0.002),
                (0.0, 0.002),
            ],
        ),
    ],
)
def test_PatientMetricAttentiveScheduler(
    base_lr, decay_factor, patience, minimum_lr, seq_loss_lr
):
    lr_scheduler = lrs.MetricAttentiveScheduler(
        learning_rate=base_lr,
        decay_factor=decay_factor,
        patience=lrs.Patience(patience, lrs.Min()),
        min_learning_rate=minimum_lr,
    )

    for loss, lr_exp in seq_loss_lr:
        lr_scheduler.step(loss)
        assert np.isclose(lr_scheduler.learning_rate, lr_exp)
