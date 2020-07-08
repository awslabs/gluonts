from typing import List, Optional, Union
import functools

import numpy as np
import mxnet as mx

from gluonts.dataset.common import DataBatch


# TODO: should the following contempate mxnet arrays or just numpy arrays?
def _is_stackable(arrays: List, axis: int = 0) -> bool:
    """
    Check if elements are scalars, have too few dimensions, or their
    target axes have equal length; i.e. they are directly `stack` able.
    """
    if isinstance(arrays[0], (mx.nd.NDArray, np.ndarray)):
        s = set(arr.shape[axis] for arr in arrays)
        return len(s) <= 1 and arrays[0].shape[axis] != 0
    return True


# TODO: should the following contempate mxnet arrays or just numpy arrays?
def _pad_arrays(
    data: List[Union[np.ndarray, mx.nd.NDArray]], axis: int = 0,
) -> List[Union[np.ndarray, mx.nd.NDArray]]:
    assert isinstance(data[0], (np.ndarray, mx.nd.NDArray))
    is_mx = isinstance(data[0], mx.nd.NDArray)

    # MxNet causes a segfault when persisting 0-length arrays. As such,
    # we add a dummy pad of length 1 to 0-length dims.
    max_len = max(1, functools.reduce(max, (x.shape[axis] for x in data)))
    padded_data = []

    for x in data:
        # MxNet lacks the functionality to pad n-D arrays consistently.
        # We fall back to numpy if x is an mx.nd.NDArray.
        if is_mx:
            x = x.asnumpy()

        pad_size = max_len - x.shape[axis]
        pad_lengths = [(0, 0)] * x.ndim
        pad_lengths[axis] = (0, pad_size)
        x_padded = np.pad(x, mode="constant", pad_width=pad_lengths)

        padded_data.append(x_padded if not is_mx else mx.nd.array(x_padded))

    return padded_data


def stack(
    data,
    ctx: Optional[mx.context.Context] = None,
    variable_length: bool = False,
):
    if variable_length and not _is_stackable(data):
        data = _pad_arrays(data, axis=0)

    if isinstance(data[0], np.ndarray):
        data = mx.nd.array(data, ctx=ctx)
    elif isinstance(data[0], (list, tuple)):
        return list(stack(t, ctx=ctx) for t in zip(*data))
    return data


def batchify(
    data: List[dict],
    ctx: Optional[mx.context.Context] = None,
    variable_length: bool = False,
) -> DataBatch:
    return {
        key: stack(
            data=[item[key] for item in data],
            ctx=ctx,
            variable_length=variable_length,
        )
        for key in data[0].keys()
    }
