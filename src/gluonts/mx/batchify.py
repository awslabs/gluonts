from typing import List, Optional

import numpy as np
import mxnet as mx

from gluonts.dataset.common import DataBatch


def stack(data, ctx: Optional[mx.context.Context] = None):
    if isinstance(data[0], np.ndarray):
        data = mx.nd.array(data, ctx=ctx)
    elif isinstance(data[0], (list, tuple)):
        return list(stack(t, ctx=ctx) for t in zip(*data))
    return data


def batchify(
    data: List[dict], ctx: Optional[mx.context.Context] = None
) -> DataBatch:
    return {
        key: stack(data=[item[key] for item in data], ctx=ctx)
        for key in data[0].keys()
    }
