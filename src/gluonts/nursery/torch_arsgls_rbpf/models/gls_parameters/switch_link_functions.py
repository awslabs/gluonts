from box import Box
from torch import nn

from torch_extensions.mlp import MLP


# class IndividualLink(nn.ModuleDict):
#     def __init__(self, dim_in, names_to_dims_out: dict, names_to_dims_hidden: dict,
#                  names_to_activations: dict):
#         super().__init__()
#         assert tuple(names_to_dims_out.keys()) == \
#                tuple(names_to_dims_hidden.keys()) == \
#                tuple(names_to_activations.keys())
#         assert all(isinstance(val, int) for val in names_to_dims_out)
#         assert all(isinstance(val, (tuple, list)) for val in names_to_dims_hidden)
#         assert all(isinstance(val, (nn.Module, tuple, list)) for val in names_to_activations)
#
#         names_to_activations = {
#             name: val if isinstance(val, (tuple, list))
#             else (names_to_activations,) * len(names_to_dims_hidden[name])
#             for name, val in names_to_activations.items()
#         }
#         for name in names_to_dims_out.keys():
#             self.update({
#                 name: MLP(
#                     dim_in=dim_in,
#                     dims_hidden=tuple(names_to_dims_hidden[name]) + (names_to_dims_out[name]),
#                     activations=tuple(names_to_activations[name]) + (nn.Softmax(dim=-1),),
#                 )
#             })
#
#     def forward(self, switch):
#         return Box({name: link(switch) for name, link in self.items()})


class IndividualLink(nn.ModuleDict):
    def __init__(self, dim_in, names_and_dims_out: dict):
        super().__init__()
        for name, dim_out in names_and_dims_out.items():
            self.update({
                name: MLP(
                    dim_in=dim_in,
                    dims_hidden=(64, dim_out,),
                    activations=(
                        nn.LeakyReLU(0.1, inplace=True), nn.Softmax(dim=-1),)
                )
            })

    def forward(self, switch):
        return Box({name: link(switch) for name, link in self.items()})


# TODO: allow this configurable. Changed back to without hidden, as KVAE uses this without.
class SharedLink(nn.Module):
    def __init__(self, dim_in, dim_out, names: (list, tuple)):
        super().__init__()
        self.names = names
        self.link = MLP(
            dim_in=dim_in,
            dims_hidden=(dim_out,),
            activations=(nn.Softmax(dim=-1),)
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
