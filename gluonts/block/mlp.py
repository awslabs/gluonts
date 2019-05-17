# Standard library imports
from typing import List

# Third-party imports
from mxnet.gluon import nn

# First-party imports
from gluonts.core.component import validated


class MLP(nn.HybridBlock):
    @validated()
    def __init__(
        self, layer_sizes: List[int], flatten: bool, activation='relu'
    ) -> None:
        super().__init__()
        self.layer_sizes = layer_sizes
        with self.name_scope():
            self.layers = nn.HybridSequential()
            for layer, layer_dim in enumerate(layer_sizes):
                self.layers.add(
                    nn.Dense(layer_dim, flatten=flatten, activation=activation)
                )

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, x):
        return self.layers(x)
