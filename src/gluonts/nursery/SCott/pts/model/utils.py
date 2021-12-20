import inspect
from typing import Optional

import torch
import torch.nn as nn


def get_module_forward_input_names(module: nn.Module):
    params = inspect.signature(module.forward).parameters
    return list(params)


def copy_parameters(net_source: nn.Module, net_dest: nn.Module) -> None:
    net_dest.load_state_dict(net_source.state_dict())


def weighted_average(
    tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None
):
    if weights is not None:
        weighted_tensor = tensor * weights
        if dim is not None:
            sum_weights = torch.sum(weights, dim)
            sum_weighted_tensor = torch.sum(weighted_tensor, dim)
        else:
            sum_weights = weights.sum()
            sum_weighted_tensor = weighted_tensor.sum()

        sum_weights = torch.max(torch.ones_like(sum_weights), sum_weights)

        return sum_weighted_tensor / sum_weights
    else:
        if dim is not None:
            return torch.mean(tensor, dim=dim)
        else:
            return tensor.mean()
