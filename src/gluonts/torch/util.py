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

from typing import List, Optional, Type
import inspect

import torch


def copy_parameters(
    net_source: torch.nn.Module,
    net_dest: torch.nn.Module,
    strict: Optional[bool] = True,
) -> None:
    """
    Copies parameters from one network to another.

    Parameters
    ----------
    net_source
        Input network.
    net_dest
        Output network.
    strict:
        whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    """

    net_dest.load_state_dict(net_source.state_dict(), strict=strict)


def get_forward_input_names(module: Type[torch.nn.Module]):
    params = inspect.signature(module.forward).parameters
    param_names = [k for k, v in params.items() if not str(v).startswith("*")]
    assert param_names[0] == "self", (
        "Expected first argument of forward to be `self`, "
        f"but found `{param_names[0]}`"
    )
    return param_names[1:]  # skip: self


def weighted_average(
    x: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None
) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given dim, masking
    values associated with weight zero,

    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Parameters
    ----------
    x
        Input tensor, of which the average must be computed.
    weights
        Weights tensor, of the same shape as `x`.
    dim
        The dim along which to average `x`

    Returns
    -------
    Tensor:
        The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(
            weights != 0, x * weights, torch.zeros_like(x)
        )
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    else:
        return x.mean(dim=dim)


def lagged_sequence_values(
    indices: List[int],
    prior_sequence: torch.Tensor,
    sequence: torch.Tensor,
) -> torch.Tensor:
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

    Returns
    -------
    Tensor
        A tensor of shape ``(N, T, L)``: if ``I = len(indices)``,
        and ``sequence.shape = (N, T, C)``, then ``L = C * I``.
    """
    assert max(indices) <= prior_sequence.shape[1], (
        f"lags cannot go further than prior sequence length, found lag"
        f" {max(indices)} while prior sequence is only"
        f"{prior_sequence.shape[1]}-long"
    )

    full_sequence = torch.cat((prior_sequence, sequence), dim=1)

    lags_values = []
    for lag_index in indices:
        begin_index = -lag_index - sequence.shape[1]
        end_index = -lag_index if lag_index > 0 else None
        lags_values.append(full_sequence[:, begin_index:end_index, ...])

    lags = torch.stack(lags_values, dim=-1)
    return lags.reshape(lags.shape[0], lags.shape[1], -1)


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        yield from self.iterable
