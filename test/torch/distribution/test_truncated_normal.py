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
from gluonts.torch.distributions import TruncatedNormal

# Mostly taken from https://github.com/pytorch/rl/blob/main/test/test_distributions.py#L127


@pytest.mark.parametrize(
    "min", [-torch.ones(3), -1, 3 * torch.tensor([-1.0, -2.0, -0.5]), -0.1]
)
@pytest.mark.parametrize(
    "max", [torch.ones(3), 1, 3 * torch.tensor([1.0, 2.0, 0.5]), 0.1]
)
@pytest.mark.parametrize(
    "vecs",
    [
        (torch.tensor([0.1, 10.0, 5.0]), torch.tensor([0.1, 10.0, 5.0])),
        (torch.zeros(7, 3), torch.ones(7, 3)),
    ],
)
@pytest.mark.parametrize(
    "upscale", [torch.ones(3), 1, 3 * torch.tensor([1.0, 2.0, 0.5]), 3]
)
@pytest.mark.parametrize("shape", [torch.Size([]), torch.Size([3, 4])])
def test_truncnormal(min, max, vecs, upscale, shape):
    torch.manual_seed(0)
    d = TruncatedNormal(
        *vecs,
        upscale=upscale,
        min=min,
        max=max,
    )
    for _ in range(100):
        a = d.rsample(shape)
        assert a.shape[: len(shape)] == shape
        assert (a >= d.min).all()
        assert (a <= d.max).all()
        lp = d.log_prob(a)
        assert torch.isfinite(lp).all()
