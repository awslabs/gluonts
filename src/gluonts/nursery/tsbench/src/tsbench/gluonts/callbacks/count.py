import numpy as np
from mxnet.gluon import nn
from .base import Callback


class ParameterCountCallback(Callback):  # type: ignore
    """
    This callback allows counting model parameters during training.

    Attributes:
        num_parameters: The number of parameters of the model. This attribute should only be
            accessed after training.
    """

    def __init__(self) -> None:
        self.num_parameters = 0

    def on_network_initialization_end(self, network: nn.HybridBlock) -> None:
        self.num_parameters = sum(
            np.prod(p.shape) for p in network.collect_params().values()
        )
