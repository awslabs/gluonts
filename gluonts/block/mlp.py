# Standard library imports
from typing import List

# Third-party imports
from mxnet.gluon import nn

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor


class MLP(nn.HybridBlock):
    """
    Defines an MLP block.

    Parameters
    ----------
    layer_sizes
        number of hidden units per layer.

    flatten
        toggle whether to flatten the output tensor.

    activation
        activation function of the MLP, default is relu.
    """
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
    def hybrid_forward(self, F, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        x
            Input tensor

        Returns
        -------
        MLP output : Tensor
            Output of the MLP given the input tensor.
        """
        return self.layers(x)
