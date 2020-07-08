from typing import List, Optional

import numpy as np
import torch

from gluonts.dataset.common import DataBatch


def stack(data, device: Optional[torch.device] = None):
    if isinstance(data[0], np.ndarray):
        data = torch.tensor(data, device=device)
    elif isinstance(data[0], (list, tuple)):
        return list(stack(t, device=device) for t in zip(*data))
    return data


def batchify(
    data: List[dict], device: Optional[torch.device] = None
) -> DataBatch:
    return {
        key: stack(data=[item[key] for item in data], device=device)
        for key in data[0].keys()
    }
