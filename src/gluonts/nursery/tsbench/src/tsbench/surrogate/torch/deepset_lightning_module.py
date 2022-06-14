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

# pylint: disable=arguments-differ,too-many-ancestors,abstract-method
from typing import List, Tuple
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping
from torch import nn, optim
from .losses import ListMLELoss


class DeepSetLightningModule(pl.LightningModule):
    """
    Lightning module which trains a deep set model until convergence.
    """

    def __init__(
        self, model: nn.Module, loss: nn.Module, weight_decay: float = 0.0
    ):
        super().__init__()

        self.model = model
        self.loss = loss
        self.weight_decay = weight_decay
        self.uses_ranking = isinstance(self.loss, ListMLELoss)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(
            self.model.parameters(), lr=1e-2, weight_decay=self.weight_decay
        )

    def configure_callbacks(self) -> List[Callback]:
        return [
            EarlyStopping(
                "train_loss",
                patience=50,
                min_delta=1e-3,
                check_on_train_epoch_end=True,
            )
        ]

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        _batch_idx: int,
    ) -> torch.Tensor:
        X, X_lengths, y_true, group_ids = batch
        y_pred = self.model(X, X_lengths)

        if self.uses_ranking:
            loss = self.loss(y_pred, y_true, group_ids)
        else:
            loss = self.loss(y_pred, y_true)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        X, X_lengths = batch
        return self.model(X, X_lengths)
