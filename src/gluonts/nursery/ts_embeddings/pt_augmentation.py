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

    def forward(self, x):
        if self.p < torch.rand(1):
            return x

        return x + torch.normal(
            mean=0.0, std=self.sigma, size=x.shape, device=x.device
        )


class Scaling(nn.Module):
    """https://arxiv.org/pdf/1706.00527.pdf"""

    def __init__(self, p, sigma=0.1):
        super().__init__()
        self.p = p
        self.sigma = sigma

    def forward(self, x):
        if self.p < torch.rand(1):
            return x
        factor = torch.normal(
            mean=1.0,
            std=self.sigma,
            size=(x.shape[0], 1, x.shape[2]),
            device=x.device,
        )
        return x * factor


class Rotation(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p < torch.rand(1):
            return x

        flip_index = torch.multinomial(
            torch.tensor([0.5, 0.5], dtype=x.dtype, device=x.device),
            num_samples=x.shape[0] * x.shape[2],
            replacement=True,
        )

        ones = torch.ones((x.shape[0] * x.shape[2]), device=x.device)
        flip = torch.where(flip_index == 0, -ones, ones)

        rotate_axis = np.arange(x.shape[2])
        np.random.shuffle(rotate_axis)

        return flip.reshape(x.shape[0], 1, x.shape[2]) * x[:, :, rotate_axis]
