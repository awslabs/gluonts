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
    prior_input: torch.Tensor,
    input: torch.Tensor,
    features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Constructs an array of lagged values from a given sequence.

    Parameters
    ----------
    indices
        Indices of the lagged observations that the RNN takes as input. For
        example, ``[0]`` indicates that the RNN only takes the observation at
        time ``t`` to produce the output for time ``t``; instead, ``[0, 24]``
        indicates that the RNN takes observations at times ``t`` and ``t-24``
        as input.
    prior_input
        Tensor containing prior input sequence.
    input
        Tensor containing input sequence for which we want lagged observations.
    features
        Tensor of additional features.

    Returns
    -------
    Tensor
        A tensor of shape ``(N, T, L)``: if ``I = len(indices)``,
        ``input.shape = (N, T, C)``, and ``features.shape = (N, T, F)``,
        then ``L = C * I + F``.
    """
    sequence = torch.cat((prior_input, input), dim=1)
    sequence_length = sequence.shape[1]

    output_sequence_length = input.shape[1]

    assert max(indices) + output_sequence_length <= sequence_length, (
        "lags cannot go further than history length, found lag"
        f" {max(indices)} while history length is only {sequence_length}"
    )

    output_values = []
    for lag_index in indices:
        begin_index = -lag_index - output_sequence_length
        end_index = -lag_index if lag_index > 0 else None
        output_values.append(sequence[:, begin_index:end_index, ...])

    output_sequence = torch.stack(output_values, dim=-1)

    lags_shape = output_sequence.shape
    reshaped_lagged_sequence = output_sequence.reshape(
        lags_shape[0], lags_shape[1], -1
    )

    if features is None:
        return reshaped_lagged_sequence
    else:
        return torch.cat((reshaped_lagged_sequence, features), dim=-1)
