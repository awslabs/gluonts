from typing import List
from mxnet import gluon
from mxnet.gluon import nn


class Callback:
    """
    A stripped-down callback which is focused on batches rather than epochs.
    """

    def on_train_start(self, trainer: gluon.Trainer) -> None:
        """
        Hook called before training is run.
        """

    def on_network_initialization_end(self, network: nn.HybridBlock) -> None:
        """
        Hook called once the network is initialized.
        """

    def on_train_batch_end(
        self, network: nn.HybridBlock, time_elapsed: float
    ) -> None:
        """
        Hook called after every training batch.
        """

    def on_validation_epoch_end(self, loss: float) -> None:
        """
        Hook called after every validation epoch.
        """


class CallbackList(Callback):
    """
    Wrapper class for a list of callbacks.
    """

    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks

    def on_train_start(self, trainer: gluon.Trainer) -> None:
        for callback in self.callbacks:
            callback.on_train_start(trainer)

    def on_network_initialization_end(self, network: nn.HybridBlock) -> None:
        for callback in self.callbacks:
            callback.on_network_initialization_end(network)

    def on_train_batch_end(
        self, network: nn.HybridBlock, time_elapsed: float
    ) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_end(network, time_elapsed)

    def on_validation_epoch_end(self, loss: float) -> None:
        for callback in self.callbacks:
            callback.on_validation_epoch_end(loss)
