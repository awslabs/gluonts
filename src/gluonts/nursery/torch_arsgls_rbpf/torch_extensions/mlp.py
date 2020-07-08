from typing import Tuple
from torch import nn
from torch_extensions.layers_with_init import Linear


class MLP(nn.Sequential):
    def __init__(
        self,
        dim_in: int,
        dims: Tuple[int],
        activations: (Tuple[nn.Module], nn.Module, None),
    ):
        super().__init__()
        assert isinstance(dims, (tuple, list))

        if not isinstance(activations, (tuple, list)):
            activations = tuple(activations for _ in range(len(dims)))

        dims_in = (dim_in,) + tuple(dims[:-1])
        dims_out = dims
        for l, (n_in, n_out, activation) in enumerate(
            zip(dims_in, dims_out, activations)
        ):
            self.add_module(name=f"linear_{l}", module=Linear(n_in, n_out))
            if activation is not None:
                self.add_module(name=f"activation_{l}", module=activation)
