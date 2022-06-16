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

from typing import List, Optional, Tuple

import torch


def lagged_sequence_values(
    indices: List[int],
    prior_sequence: torch.Tensor,
    sequence: torch.Tensor,
    features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Constructs an array of lagged values from a given sequence.

    Parameters
    ----------
    indices
        Indices of the lagged observations. For example, ``[0]`` indicates
        that, at any time ``t``, the will have only the observation from
        time ``t`` itself; instead, ``[0, 24]`` indicates that the output
        will have observations from times ``t`` and ``t-24``.
    prior_sequence
        Tensor containing the input sequence prior to the time range for
        which the output is required (shape: ``(N, H, C)``).
    sequence
        Tensor containing the input sequence in the time range where the
        output is required (shape: ``(N, T, C)``).
    features
        Tensor of additional features.

    Returns
    -------
    Tensor
        A tensor of shape ``(N, T, L)``: if ``I = len(indices)``,
        ``sequence.shape = (N, T, C)``, and ``features.shape = (N, T, F)``,
        then ``L = C * I + F``.
    """
    assert max(indices) <= prior_sequence.shape[1], (
        f"lags cannot go further than prior sequence length, found lag"
        f" {max(indices)} while prior sequence is only"
        f"{prior_sequence.shape[1]}-long"
    )

    full_sequence = torch.cat((prior_sequence, sequence), dim=1)

    output_values = []
    for lag_index in indices:
        begin_index = -lag_index - sequence.shape[1]
        end_index = -lag_index if lag_index > 0 else None
        output_values.append(full_sequence[:, begin_index:end_index, ...])

    output_sequence = torch.stack(output_values, dim=-1)

    lags_shape = output_sequence.shape
    reshaped_lagged_sequence = output_sequence.reshape(
        lags_shape[0], lags_shape[1], -1
    )

    if features is None:
        return reshaped_lagged_sequence
    else:
        return torch.cat((reshaped_lagged_sequence, features), dim=-1)
