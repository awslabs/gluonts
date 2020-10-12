import torch.nn as nn


class LambdaLayer(nn.Module):
    def __init__(self, function):
        super().__init__()
        self._func = function

    def forward(self, x, *args):
        return self._func(x, *args)
