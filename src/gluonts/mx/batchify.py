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

import functools
from typing import List, Optional, Type

import mxnet as mx
import numpy as np

from gluonts.dataset.common import DataBatch


def pad_to_size(
    x: np.array, size: int, axis: int = 0, is_right_pad: bool = True
):
    """
    Pads `xs` with 0 on the right (default) on the specified axis, which is the
    first axis by default.
    """

    pad_length = size - x.shape[axis]
    if pad_length <= 0:
        return x

    pad_width = [(0, 0)] * x.ndim
    right_pad = (0, pad_length)
    pad_width[axis] = right_pad if is_right_pad else right_pad[::-1]
    return np.pad(x, mode="constant", pad_width=pad_width)


def _is_stackable(arrays: List, axis: int = 0) -> bool:
    """
    Check if elements are scalars, have too few dimensions, or their target
    axes have equal length; i.e. they are directly `stack` able.
    """
    if isinstance(arrays[0], np.ndarray):
        s = {arr.shape[axis] for arr in arrays}
        return len(s) <= 1 and arrays[0].shape[axis] != 0
    return True


def _pad_arrays(
    data: List[np.ndarray],
    axis: int = 0,
    is_right_pad: bool = True,
) -> List[np.ndarray]:
    assert isinstance(data[0], np.ndarray)

    # MxNet causes a segfault when persisting 0-length arrays. As such,
    # we add a dummy pad of length 1 to 0-length dims.
    max_len = max(1, functools.reduce(max, (x.shape[axis] for x in data)))
    return [
        pad_to_size(x, max_len, axis, is_right_pad=is_right_pad) for x in data
    ]


def stack(
    data,
    ctx: Optional[mx.context.Context] = None,
    dtype: Optional[Type] = np.float32,
    variable_length: bool = False,
    is_right_pad: bool = True,
):
    if variable_length and not _is_stackable(data):
        data = _pad_arrays(data, axis=0, is_right_pad=is_right_pad)
    if isinstance(data[0], mx.nd.NDArray):
        # TODO: think about using shared context NDArrays
        #  https://github.com/awslabs/gluonts/blob/42bee73409f801e7bca73245ca21cd877891437c/src/gluonts/dataset/parallelized_loader.py#L157
        return mx.nd.stack(*data)
    if isinstance(data[0], np.ndarray):
        data = mx.nd.array(data, dtype=dtype, ctx=ctx)
    elif isinstance(data[0], (list, tuple)):
        return list(
            stack(
                t,
                ctx=ctx,
                dtype=dtype,
                variable_length=variable_length,
                is_right_pad=is_right_pad,
            )
            for t in zip(*data)
        )
    return data


def batchify(
    data: List[dict],
    ctx: Optional[mx.context.Context] = None,
    dtype: Optional[Type] = np.float32,
    variable_length: bool = False,
    is_right_pad: bool = True,
) -> DataBatch:
    return {
        key: stack(
            data=[item[key] for item in data],
            ctx=ctx,
            dtype=dtype,
            variable_length=variable_length,
            is_right_pad=is_right_pad,
        )
        for key in data[0].keys()
    }


def as_in_context(batch: dict, ctx: mx.Context = None) -> DataBatch:
    """
    Move data into new context, should only be in main process.
    """
    batch = {
        k: v.as_in_context(ctx) if isinstance(v, mx.nd.NDArray)
        # Workaround due to MXNet not being able to handle NDArrays with 0 in
        # shape properly:
        else (
            stack(v, ctx=ctx, dtype=v.dtype, variable_length=False)
            if isinstance(v[0], np.ndarray) and 0 in v[0].shape
            else v
        )
        for k, v in batch.items()
    }
    return batch
