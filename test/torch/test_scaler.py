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

import torch
import pytest

from gluonts.torch import scaler

test_cases = [
    (
        scaler.MeanScaler(),
        torch.Tensor(
            [
                [1.0] * 50,
                [0.0] * 25 + [3.0] * 25,
                [2.0] * 49 + [1.5] * 1,
                [0.0] * 50,
                [1.0] * 50,
            ]
        ),
        torch.Tensor(
            [
                [1.0] * 50,
                [0.0] * 25 + [1.0] * 25,
                [0.0] * 49 + [1.0] * 1,
                [1.0] * 50,
                [0.0] * 50,
            ]
        ),
        torch.Tensor([1.0, 3.0, 1.5, 1e-10, 1.00396824]),
    ),
    (
        scaler.MeanScaler(default_scale=0.5),
        torch.Tensor(
            [
                [1.0] * 50,
                [0.0] * 25 + [3.0] * 25,
                [2.0] * 49 + [1.5] * 1,
                [0.0] * 50,
                [1.0] * 50,
            ]
        ),
        torch.Tensor(
            [
                [0.0] * 50,
                [0.0] * 25 + [1.0] * 25,
                [0.0] * 49 + [1.0] * 1,
                [1.0] * 50,
                [0.0] * 50,
            ]
        ),
        torch.Tensor([0.5, 3.0, 1.5, 1e-10, 0.5]),
    ),
    (
        scaler.MeanScaler(keepdim=True),
        torch.Tensor(
            [
                [1.0] * 50,
                [0.0] * 25 + [3.0] * 25,
                [2.0] * 49 + [1.5] * 1,
                [0.0] * 50,
                [1.0] * 50,
            ]
        ),
        torch.Tensor(
            [
                [1.0] * 50,
                [0.0] * 25 + [1.0] * 25,
                [0.0] * 49 + [1.0] * 1,
                [1.0] * 50,
                [0.0] * 50,
            ]
        ),
        torch.Tensor([1.0, 3.0, 1.5, 1e-10, 1.00396824]).unsqueeze(1),
    ),
    (
        scaler.MeanScaler(),
        torch.Tensor(
            [
                [120.0] * 25 + [150.0] * 25,
                [0.0] * 10 + [3.0] * 20 + [61.0] * 20,
                [0.0] * 50,
                [2e-2] * 10 + [0.0] * 30 + [3e-2] * 10,
            ]
        ),
        torch.Tensor(
            [
                [1.0] * 25 + [1.0] * 25,
                [0.0] * 10 + [1.0] * 20 + [1.0] * 20,
                [0.0] * 50,
                [1.0] * 10 + [0.0] * 30 + [1.0] * 10,
            ]
        ),
        torch.Tensor([135.0, 32.0, 73.00454712, 2.5e-2]),
    ),
    (
        scaler.MeanScaler(),
        torch.randn(size=(5, 30)),
        torch.zeros(size=(5, 30)),
        1e-10 * torch.ones(size=(5,)),
    ),
]


@pytest.mark.parametrize("s, target, observed, expected_scale", test_cases)
def test_scaler(s, target, observed, expected_scale):
    target_scaled, _, scale = s(target, observed)

    assert torch.allclose(
        expected_scale, scale
    ), "mismatch in the scale computation"

    if s.keepdim:
        expected_target_scaled = torch.div(target, expected_scale)
    else:
        expected_target_scaled = torch.div(
            target, expected_scale.unsqueeze(s.dim)
        )

    assert torch.allclose(
        expected_target_scaled, target_scaled
    ), "mismatch in the scaled target computation"


@pytest.mark.parametrize(
    "target, observed",
    [
        (
            torch.Tensor(
                [
                    [1.0] * 50,
                    [0.0] * 25 + [3.0] * 25,
                    [2.0] * 49 + [1.5] * 1,
                    [0.0] * 50,
                    [1.0] * 50,
                ]
            ),
            torch.Tensor(
                [
                    [1.0] * 50,
                    [0.0] * 25 + [1.0] * 25,
                    [0.0] * 49 + [1.0] * 1,
                    [1.0] * 50,
                    [0.0] * 50,
                ]
            ),
        )
    ],
)
def test_nopscaler(target, observed):
    s = scaler.NOPScaler()
    target_scaled, loc, scale = s(target, observed)

    assert torch.allclose(torch.zeros(target.shape[:-1]), loc)
    assert torch.allclose(target, target_scaled)
    assert torch.allclose(torch.ones(target.shape[:-1]), scale)
