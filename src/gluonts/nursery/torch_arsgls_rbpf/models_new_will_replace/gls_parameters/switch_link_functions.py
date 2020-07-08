from typing import Dict, Sequence, Tuple
from box import Box
from torch import nn

from torch_extensions.mlp import MLP


class IndividualLink(nn.ModuleDict):
    def __init__(
            self,
            dim_in,
            names_and_dims_out: Dict[str, int],
            dims_hidden: Tuple[int] = tuple(),
            activations_hidden: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        if isinstance(activations_hidden, nn.Module):
            activations_hidden = (activations_hidden,) * len(dims_hidden)
        for name, dim_out in names_and_dims_out.items():
            self.update(
                {
                    name: MLP(
                        dim_in=dim_in,
                        dims=dims_hidden + (dim_out,),
                        activations=activations_hidden + (nn.Softmax(dim=-1),),
                    )
                }
            )

    def forward(self, switch):
        return Box({name: link(switch) for name, link in self.items()})


class SharedLink(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_out,
            names: (list, tuple),
            dims_hidden: Tuple[int] = tuple(),
            activations_hidden: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        if isinstance(activations_hidden, nn.Module):
            activations_hidden = (activations_hidden,) * len(dims_hidden)
        self.names = names
        self.link = MLP(
            dim_in=dim_in,
            dims=dims_hidden + (dim_out,),
            activations=activations_hidden + (nn.Softmax(dim=-1),),
        )

    def forward(self, switch):
        output = self.link(switch)
        return Box({name: output for name in self.names})


class IdentityLink(nn.Module):
    def __init__(self, names):
        super().__init__()
        self.names = names

    def forward(self, switch):
        return Box({name: switch for name in self.names})
