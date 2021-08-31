import numpy as np

import torch
import torch.nn as nn


class RandomApply(nn.Module):
    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.p = p

    def forward(self, x):
        if self.p < torch.rand(1):
            return x

        for t in self.transforms:
            x = t(x)
        return x


class Jitter(nn.Module):
    """https://arxiv.org/pdf/1706.00527.pdf"""

    def __init__(self, p, sigma=0.03):
        super().__init__()
        self.p = p
        self.sigma = sigma

    def __call__(self, x):
        if self.p < torch.rand(1):
            return x

        return x + torch.normal(mean=0.0, std=self.sigma, size=x.shape)


class Scaling(nn.Module):
    """https://arxiv.org/pdf/1706.00527.pdf"""

    def __init__(self, p, sigma=0.1):
        super().__init__()
        self.p = p
        self.sigma = sigma

    def __call__(self, x):
        if self.p < torch.rand(1):
            return x
        factor = torch.normal(
            mean=1.0, std=self.sigma, size=(x.shape[0], 1, x.shape[2])
        )
        return x * factor
