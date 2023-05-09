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
    dim: int,
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
        which the output is required.
    sequence
        Tensor containing the input sequence in the time range where the
        output is required.
    dim
        Time dimension.

    Returns
    -------
    Tensor
        A tensor of shape (*sequence.shape, len(indices)).
    """
    assert max(indices) <= prior_sequence.shape[dim], (
        f"lags cannot go further than prior sequence length, found lag"
        f" {max(indices)} while prior sequence is only"
        f" {prior_sequence.shape[dim]}-long"
    )

    full_sequence = torch.cat((prior_sequence, sequence), dim=dim)

    lags_values = []
    for lag_index in indices:
        begin_index = -lag_index - sequence.shape[dim]
        end_index = -lag_index if lag_index > 0 else None
        lags_values.append(
            slice_along_dim(
                full_sequence, dim=dim, slice_=slice(begin_index, end_index)
            )
        )

    return torch.stack(lags_values, dim=-1)


def repeat_along_dim(a: torch.Tensor, dim: int, repeats: int) -> torch.Tensor:
    """
    Repeat a tensor along a given dimension, using ``torch.repeat`` internally.

    Parameters
    ----------
    a
        Original tensor to repeat.
    dim
        Dimension to repeat data over.
    repeats
        How many time to repeat the input tensor.

    Returns
    -------
    torch.Tensor
        A tensor with the same size as the input one, except dimension
        ``dim`` which is multiplied by ``repeats``.
    """
    if repeats == 1:
        return a
    r = [1] * len(a.shape)
    r[dim] = repeats
    return a.repeat(*r)


def slice_along_dim(a: torch.Tensor, dim: int, slice_: slice) -> torch.Tensor:
    """
    Slice a tensor along a given dimension.

    Parameters
    ----------
    a
        Original tensor to slice.
    dim
        Dimension to slice over.
    slice_
        Slice to take.

    Returns
    -------
    torch.Tensor
        A tensor with the same size as the input one, except dimension
        ``dim`` which has length equal to the slice length.
    """
    idx = [slice(None)] * len(a.shape)
    idx[dim] = slice_
    return a[idx]


def take_last(a: torch.Tensor, dim: int, num: int) -> torch.Tensor:
    """
    Take last elements from a given tensor along a given dimension.

    Parameters
    ----------
    a
        Original tensor to slice.
    dim
        Dimension to slice over.
    num
        Number of trailing elements to retain (non-negative).

    Returns
    -------
    torch.Tensor
        A tensor with the same size as the input one, except dimension
        ``dim`` which has length equal to ``num``.
    """
    assert num >= 0
    return slice_along_dim(a, dim, slice(a.shape[dim] - num, None))


def unsqueeze_expand(a: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """
    Unsqueeze a dimension and expand over it in one go.

    Parameters
    ----------
    a
        Original tensor to unsqueeze.
    dim
        Dimension to unsqueeze.
    size
        Size for the new dimension.

    Returns
    -------
    torch.Tensor
        A tensor with an added dimension ``dim`` of size ``size``.
    """
    a = a.unsqueeze(dim)
    sizes = list(a.shape)
    sizes[dim] = size
    return a.expand(*sizes)
