import torch
from torch import nn


class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.exp(input)
