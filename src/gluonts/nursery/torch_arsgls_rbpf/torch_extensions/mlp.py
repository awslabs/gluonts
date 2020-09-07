from enum import Enum
from typing import Optional
from typing import Tuple
from torch import nn
from torch_extensions.layers_with_init import Linear


class NormalizationType(Enum):
    none = 1
    layer = 2
    layer_learnable = 3
    batch = 4
    batch_learnable = 5


def make_norm_layer(norm_type, shp_features):
    if norm_type is None or norm_type.value == NormalizationType.none.value:
        return None
    elif norm_type.value == NormalizationType.layer.value:
        return nn.LayerNorm(shp_features, elementwise_affine=False)
    elif norm_type.value == NormalizationType.layer_learnable.value:
        return nn.LayerNorm(shp_features, elementwise_affine=True)
    else:
        raise NotImplementedError(f"did not implement '{norm_type}'")


class MLP(nn.Sequential):
    def __init__(
        self,
        dim_in: int,
        dims: Tuple[int],
        activations: (Tuple[nn.Module], nn.Module, None),
        norm_type: Optional[NormalizationType] = None,
        # norm_type: Optional[NormalizationType] = NormalizationType.layer_learnable,
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
            norm_layer = make_norm_layer(
                norm_type=norm_type, shp_features=[n_out],  # after Dense -> out
            )
            if norm_layer is not None:
                self.add_module(name=f"norm_{norm_type.name}_{l}", module=norm_layer)
            if activation is not None:
                self.add_module(name=f"activation_{l}", module=activation)
