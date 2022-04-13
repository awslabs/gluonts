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

from pathlib import Path
from typing import List, Optional
from mxnet.gluon import nn
from mxnet.gluon.trainer import Trainer
from .base import Callback


class ModelSaverCallback(Callback):  # type: ignore
    """
    The model saver callback saves the model during training at exponential
    frequency.

    Attributes:
        network: The network that was trained. Not available prior to training.
        saved_parameters: The parameters saved for the different milestones. Should only be
            accessed after training has finished and should not be modified.
        training_times: The training times in seconds for the different milestones.
        num_gradient_updates: The number of gradient updates for the different milestones.
    """

    def __init__(
        self,
        directory: Path,
        milestones: List[float],
    ):
        """
        Args:
            directory: The directory into which the model parameters ought to be saved. Models are
                stored as `model_<seq>.params` where `<seq>` starts with 0 and reaches
                `count - 1`. Note that the directory must exist.
            milestones: The number of seconds after which the model should be saved.
        """
        assert all(
            x < y for x, y in zip(milestones, milestones[1:])
        ), "Time milestones must be increasing."

        super().__init__()
        self.directory = directory
        self.seq = 0
        self.batch_count = 0
        self.network: Optional[nn.HybridBlock] = None
        self.saved_parameters: List[Path] = []
        self.training_times: List[float] = []
        self.num_gradient_updates: List[int] = []
        self.milestones = milestones

    def on_train_start(self, trainer: Trainer) -> None:
        self.seq = 0
        self.batch_count = 0
        self.network = None
        self.saved_parameters = []
        self.training_times = []
        self.num_gradient_updates = []

    def on_network_initialization_end(self, network: nn.HybridBlock) -> None:
        self.network = network

    def on_train_batch_end(
        self, network: nn.HybridBlock, time_elapsed: float
    ) -> None:
        self.batch_count += 1
        if (
            len(self.milestones) > self.seq
            and time_elapsed > self.milestones[self.seq]
        ):
            file = self.directory / f"model_{self.seq}.params"
            network.save_parameters(file.absolute().as_posix())
            self.saved_parameters.append(file)
            self.training_times.append(time_elapsed)
            self.num_gradient_updates.append(self.batch_count)
            self.seq += 1
