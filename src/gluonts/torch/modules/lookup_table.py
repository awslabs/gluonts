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
import torch.nn as nn

from gluonts.core.component import validated


class LookupValues(nn.Module):
    """
    A lookup table mapping bin indices to values.

    Parameters
    ----------
    bin_values
        Tensor of bin values with shape (num_bins, ).
    """

    @validated()
    def __init__(self, bin_values: torch.Tensor):
        super().__init__()
        self.register_buffer("bin_values", bin_values)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.bin_values, torch.Tensor)
        indices = torch.clamp(indices, 0, self.bin_values.shape[0] - 1)
        return torch.index_select(
            self.bin_values, 0, indices.reshape(-1)
        ).view_as(indices)
