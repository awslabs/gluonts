from typing import Optional, Union, Sequence
import torch
from torch import nn


class Constant(nn.Module):
    """
    Module that outputs a constant with the batch shape corresponding to inputs.
    In case of 0, use torch.zeros explicitly otherwise val * torch.ones.
    """
    def __init__(
            self,
            val: Union[int, float, torch.Tensor],
            shp_append: Sequence[int],
            n_dims_from_input: Optional[int] = -1,
    ):
        super().__init__()
        self.val = val
        self.shp_append = tuple(shp_append)
        self.n_dims_from_input = n_dims_from_input

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        self._val = val
        if val == 0:
            self.forward = self._forward_zeros
        else:
            self.forward = self._forward_const

    def _forward_const(self, x: torch.Tensor):
        shp = x.shape[:self.n_dims_from_input] + self.shp_append
        return self.val * torch.ones(shp, device=x.device, dtype=x.dtype)

    def _forward_zeros(self, x: torch.Tensor):
        shp = x.shape[:self.n_dims_from_input] + self.shp_append
        return torch.zeros(shp, device=x.device, dtype=x.dtype)
