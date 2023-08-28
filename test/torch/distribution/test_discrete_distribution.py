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

import pytest

import torch

from gluonts.torch.distributions import DiscreteDistribution


@pytest.mark.parametrize(
    "values, probs",
    [
        # Both elements in the batch have same distribution parameters.
        (
            torch.tensor([[0.0, 1.0, 0.5], [0.0, 0.5, 1.0]]),
            torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]]),
        )
    ],
)
@pytest.mark.parametrize(
    "obs, rps",
    [
        (
            torch.tensor([[0.0], [1.0]]),
            torch.tensor([0.0000, 0.2500]),
            # (0.5 * (0.0 - 0.0) + 0.0 + (1.0 - 1.0) * (1.0 - 0.5)) / 2
            # (0.5 * (1.0 - 0.0) + 0.0 + 1.0 * (1.0 - 1.0)) / 2
        ),
        (
            torch.tensor([[0.5], [0.75]]),
            torch.tensor([0.1250, 0.1875]),
            # (0.5 * (0.5 - 0.0) + 0.0 + (1.0 - 1.0) * (1.0 - 0.5)) / 2
            # (0.5 * (0.75 - 0.0) + 0.0 + (1.0 - 1.0) * (1.0 - 0.75)) / 2
        ),
        (
            torch.tensor([[-10.0], [10.0]]),
            torch.tensor([2.5, 7.0]),
            # (0.5 * (0.0 + 10.0) + 0.0 + (1.0 - 1.0) * (1.0 + 10.0)) / 2
            # (0.5 * (10.0 - 0.0) + 0.0 + (1.0) * (10.0 - 1.0)) / 2
        ),
    ],
)
def test_rps(values, probs, obs, rps):
    discrete_distr = DiscreteDistribution(values=values, probs=probs)
    rps_computed = discrete_distr.rps(obs)
    assert all(rps == rps_computed)


@pytest.mark.parametrize(
    "values, probs",
    [
        # Duplicate values occur (i) only in the middle (ii) at the extremes
        (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 5.0],
                    [-1.0, -1.0, 0.0, 0.0, 2.0, 5.0, 5.0],
                ]
            ),
            torch.tensor(
                [
                    [0.1, 0.12, 0.03, 0.15, 0.05, 0.15, 0.4],
                    [0.15, 0.05, 0.13, 0.12, 0.05, 0.27, 0.23],
                ]
            ),
        )
    ],
)
@pytest.mark.parametrize(
    "probs_adjusted",
    [
        torch.tensor(
            [
                [0.1, 0.0, 0.0, 0.3, 0.0, 0.2, 0.4],
                [0.0, 0.2, 0.0, 0.25, 0.05, 0.0, 0.5],
            ]
        ),
    ],
)
def test_probs_duplicate_values(values, probs, probs_adjusted):
    probs_adjusted_computed = DiscreteDistribution.adjust_probs(values, probs)
    assert all(probs_adjusted.view(-1) == probs_adjusted_computed.view(-1))
