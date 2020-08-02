from typing import Union
import torch
from torch import nn


class Affine(nn.Module):
    def __init__(
            self,
            loc: Union[float, torch.Tensor],
            scale: Union[float, torch.Tensor],
    ):
        super().__init__()
        self.loc = loc
        self.scale = scale

    def forward(self, x: torch.Tensor):
        return self.loc + self.scale * x


class Bias(nn.Module):
    def __init__(
            self,
            loc: Union[float, torch.Tensor],
    ):
        super().__init__()
        self.loc = loc

    def forward(self, x: torch.Tensor):
        return self.loc + x


class Factor(nn.Module):
    def __init__(
            self,
            scale: Union[float, torch.Tensor],
    ):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor):
        return self.scale * x
