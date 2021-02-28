from typing import List, Tuple

import torch as pt
from torch import Tensor
from torch import nn
from torch.distributions import Distribution

from .activations import PositiveSoftplus


def distribution_cat(distributions: List[Distribution], dim=0) -> Distribution:
    args = list(distributions[0].arg_constraints.keys())
    if "probs" in args and "logits" in args:
        try:
            _ = getattr(distributions[0], "probs")
            args.remove("logits")
        except AttributeError:
            args.remove("probs")
    concat_args = {
        arg: pt.cat([getattr(d, arg) for d in distributions], dim=dim)
        for arg in args
    }
    return type(distributions[0])(**concat_args)


class GaussianLayer(nn.Module):
    def __init__(self, d_hidden: int, d_data: int):
        super().__init__()
        self.mean = nn.Linear(d_hidden, d_data)
        self.var = nn.Sequential(
            nn.Linear(d_hidden, d_data), PositiveSoftplus(1e-3)
        )

    def forward(self, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        mu = self.mean(hidden)
        sigma = self.var(hidden)
        return mu, sigma
