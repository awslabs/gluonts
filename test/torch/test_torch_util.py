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

from typing import List

import pytest
import torch

from gluonts.torch.util import (
    lagged_sequence_values,
    unsqueeze_expand,
)


@pytest.mark.parametrize(
    "lag_indices, prior_sequence, sequence, output_shape",
    [
        (
            [0, 1, 5, 10, 20],
            torch.randn((4, 100)),
            torch.randn((4, 8)),
            (4, 8, 5),
        ),
        (
            [0, 1, 5, 10, 20],
            torch.randn((4, 100, 1)),
            torch.randn((4, 8, 1)),
            (4, 8, 1, 5),
        ),
        (
            [0, 1, 5, 10, 20],
            torch.randn((4, 100, 2)),
            torch.randn((4, 8, 2)),
            (4, 8, 2, 5),
        ),
    ],
)
def test_lagged_sequence_values(
    lag_indices: List[int],
    prior_sequence: torch.Tensor,
    sequence: torch.Tensor,
    output_shape: tuple,
):
    res = lagged_sequence_values(lag_indices, prior_sequence, sequence, dim=1)
    assert res.shape == output_shape
    full_sequence = torch.cat((prior_sequence, sequence), dim=1)
    for t in range(res.shape[1]):
        expected_lags_t = torch.stack(
            [
                full_sequence[:, t + prior_sequence.shape[1] - l]
                for l in lag_indices
            ],
            dim=-1,
        )
        assert expected_lags_t.shape == res[:, t, :].shape
        assert torch.allclose(expected_lags_t, res[:, t, :])


@pytest.mark.parametrize(
    "input_shape, dim, size, output_shape",
    [
        ((3, 4, 5), 2, 42, (3, 4, 42, 5)),
        ((3, 4, 5), 0, 42, (42, 3, 4, 5)),
        ((3, 4, 5), -1, 42, (3, 4, 5, 42)),
        ((3, 4, 5), -2, 42, (3, 4, 42, 5)),
    ],
)
def test_unsqueeze_expand(input_shape, dim, size, output_shape):
    a = torch.randn(size=input_shape)
    b = unsqueeze_expand(a, dim=dim, size=size)
    assert b.shape == output_shape
