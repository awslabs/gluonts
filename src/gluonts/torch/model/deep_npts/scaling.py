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

from typing import Tuple
import torch


def _one_if_too_small(x: torch.Tensor, min_value) -> torch.Tensor:
    return torch.where(
        x >= min_value, x, torch.ones(tuple(1 for _ in x.shape), dtype=x.dtype)
    )


def min_max_scaling(
    seq: torch.Tensor, dim=-1, keepdim=False, min_scale=1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    loc = torch.min(seq, dim=dim, keepdim=keepdim)[0]
    scale = torch.max(seq, dim=dim, keepdim=keepdim)[0] - loc
    return loc, _one_if_too_small(scale, min_value=min_scale)


def standard_normal_scaling(
    seq: torch.Tensor, dim=-1, keepdim=False, min_scale=1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale, loc = torch.std_mean(seq, dim=dim, keepdim=keepdim)
    return loc, _one_if_too_small(scale, min_value=min_scale)
