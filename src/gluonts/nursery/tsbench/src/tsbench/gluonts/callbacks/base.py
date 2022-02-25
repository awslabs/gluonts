# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

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
