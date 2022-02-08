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
import torch

from pts.modules import MeanScaler, NOPScaler

test_cases = [
    (
        MeanScaler(),
        torch.tensor(
            [
                [1.0] * 50,
                [0.0] * 25 + [3.0] * 25,
                [2.0] * 49 + [1.5] * 1,
                [0.0] * 50,
                [1.0] * 50,
            ]
        ),
        torch.tensor(
            [
                [1.0] * 50,
                [0.0] * 25 + [1.0] * 25,
                [0.0] * 49 + [1.0] * 1,
                [1.0] * 50,
                [0.0] * 50,
            ]
        ),
        torch.tensor([1.0, 3.0, 1.5, 1.00396824, 1.00396824]),
    ),
    (
        MeanScaler(keepdim=True),
        torch.tensor(
            [
                [1.0] * 50,
                [0.0] * 25 + [3.0] * 25,
                [2.0] * 49 + [1.5] * 1,
                [0.0] * 50,
                [1.0] * 50,
            ]
        ),
        torch.tensor(
            [
                [1.0] * 50,
                [0.0] * 25 + [1.0] * 25,
                [0.0] * 49 + [1.0] * 1,
                [1.0] * 50,
                [0.0] * 50,
            ]
        ),
        torch.tensor([1.0, 3.0, 1.5, 1.00396824, 1.00396824]).unsqueeze(1),
    ),
    (
        MeanScaler(),
        torch.tensor(
            [
                [[1.0]] * 50,
                [[0.0]] * 25 + [[3.0]] * 25,
                [[2.0]] * 49 + [[1.5]] * 1,
                [[0.0]] * 50,
                [[1.0]] * 50,
            ]
        ),
        torch.tensor(
            [
                [[1.0]] * 50,
                [[0.0]] * 25 + [[1.0]] * 25,
                [[0.0]] * 49 + [[1.0]] * 1,
                [[1.0]] * 50,
                [[0.0]] * 50,
            ]
        ),
        torch.tensor([1.0, 3.0, 1.5, 1.00396824, 1.00396824]).unsqueeze(1),
    ),
    (
        MeanScaler(minimum_scale=1e-8),
        torch.tensor(
            [
                [[1.0, 2.0]] * 50,
                [[0.0, 0.0]] * 25 + [[3.0, 6.0]] * 25,
                [[2.0, 4.0]] * 49 + [[1.5, 3.0]] * 1,
                [[0.0, 0.0]] * 50,
                [[1.0, 2.0]] * 50,
            ]
        ),
        torch.tensor(
            [
                [[1.0, 1.0]] * 50,
                [[0.0, 1.0]] * 25 + [[1.0, 0.0]] * 25,
                [[1.0, 0.0]] * 49 + [[0.0, 1.0]] * 1,
                [[1.0, 0.0]] * 50,
                [[0.0, 1.0]] * 50,
            ]
        ),
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 1.61111116],
                [2.0, 3.0],
                [1.28160918, 1.61111116],
                [1.28160918, 2.0],
            ]
        ),
    ),
    (
        MeanScaler(),
        torch.tensor(
            [
                [120.0] * 25 + [150.0] * 25,
                [0.0] * 10 + [3.0] * 20 + [61.0] * 20,
                [0.0] * 50,
                [2e-2] * 10 + [0.0] * 30 + [3e-2] * 10,
            ]
        ),
        torch.tensor(
            [
                [1.0] * 25 + [1.0] * 25,
                [0.0] * 10 + [1.0] * 20 + [1.0] * 20,
                [0.0] * 50,
                [1.0] * 10 + [0.0] * 30 + [1.0] * 10,
            ]
        ),
        torch.tensor([135.0, 32.0, 73.00454712, 2.5e-2]),
    ),
    (
        MeanScaler(),
        torch.randn((5, 30)),
        torch.zeros((5, 30)),
        1e-10 * torch.ones((5,)),
    ),
    (
        MeanScaler(minimum_scale=1e-6),
        torch.randn((5, 30, 1)),
        torch.zeros((5, 30, 1)),
        1e-6 * torch.ones((5, 1)),
    ),
    (
        MeanScaler(minimum_scale=1e-12),
        torch.randn((5, 30, 3)),
        torch.zeros((5, 30, 3)),
        1e-12 * torch.ones((5, 3)),
    ),
    (
        NOPScaler(),
        torch.randn((10, 20, 30)),
        torch.randn((10, 20, 30)) > 0,
        torch.ones((10, 30)),
    ),
    (
        NOPScaler(),
        torch.randn((10, 20, 30)),
        torch.ones((10, 20, 30)),
        torch.ones((10, 30)),
    ),
    (
        NOPScaler(),
        torch.randn((10, 20, 30)),
        torch.zeros((10, 20, 30)),
        torch.ones((10, 30)),
    ),
]


@pytest.mark.parametrize("s, target, observed, expected_scale", test_cases)
def test_scaler(s, target, observed, expected_scale):
    target_scaled, scale = s(target, observed)

    assert np.allclose(
        expected_scale.numpy(), scale.numpy()
    ), "mismatch in the scale computation"

    if s.keepdim:
        expected_target_scaled = target / expected_scale
    else:
        expected_target_scaled = target / expected_scale.unsqueeze(1)

    assert np.allclose(
        expected_target_scaled.numpy(), target_scaled.numpy()
    ), "mismatch in the scaled target computation"


@pytest.mark.parametrize("target, observed", [])
def test_nopscaler(target, observed):
    s = NOPScaler()
    target_scaled, scale = s(target, observed)

    assert torch.norm(target - target_scaled) == 0
    assert torch.norm(torch.ones_like(target).mean(dim=1) - scale) == 0
