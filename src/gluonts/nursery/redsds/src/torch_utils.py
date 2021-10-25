from typing import Tuple
import torch
import torch.nn as nn


def unravel_indices(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


def init_layers(module):
    print(module)
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif (type(module) == nn.RNNCell
          or type(module) == nn.RNN
          or type(module) == nn.GRUCell
          or type(module) == nn.GRU
          or type(module) == nn.LSTMCell
          or type(module) == nn.LSTM):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            else:
                nn.init.xavier_uniform_(param)


def torch2numpy(tensor):
    arr = tensor.data.cpu().numpy()
    return arr
