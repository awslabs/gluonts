import numpy as np
import torch
from torchvision.transforms import Compose
from torch.utils.data._utils.collate import default_collate


class BatchFirstToTimeFirst(object):
    def __call__(self, item: dict):
        assert isinstance(item, dict)
        transformed_item = dict()

        for name, val in item.items():
            if isinstance(val, np.ndarray):
                transformed_val = val.transpose(
                    (1, 0,) + tuple(np.arange(2, val.ndim))
                )
            elif isinstance(val, torch.Tensor):
                transformed_val = val.transpose(1, 0)
            else:
                raise ValueError(
                    f"Unexpected type for value in: {name}. Got {type(val)}."
                )
            transformed_item.update({name: transformed_val})
        return transformed_item


time_first_collate_fn = Compose(
    transforms=[default_collate, BatchFirstToTimeFirst()]
)
