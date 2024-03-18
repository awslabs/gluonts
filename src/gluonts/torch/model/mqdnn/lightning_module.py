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

import pytorch_lightning as pl
import torch

from gluonts.torch.util import weighted_average

from .module import MQDNNModel


class MQDNNLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: MQDNNModel,
        quantiles: List[int],
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.quantiles = torch.tensor(quantiles)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _compute_loss(self, batch):
        output = self.model(
            past_target=batch["past_target"],
            past_observed_values=batch.get("past_observed_values", None),
            past_feat_dynamic=batch.get("past_feat_dynamic", None),
            future_feat_dynamic=batch.get("future_feat_dynamic", None),
            feat_static_real=batch.get("feat_static_real", None),
            feat_static_cat=batch.get("feat_static_cat", None),
        )

        future_target = batch["future_target"]
        observed_values = batch.get("future_observed_values", None)

        error = future_target.unsqueeze(1) - output
        pos_error = (error > 0) * error
        neg_error = (error <= 0) * error

        q = self.quantiles.unsqueeze(0).unsqueeze(2)

        loss_values = (q * pos_error + (q - 1) * neg_error).mean(axis=1)
        return weighted_average(loss_values, weights=observed_values)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        train_loss = self._compute_loss(batch)
        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute validation step.
        """
        val_loss = self._compute_loss(batch)
        self.log(
            "val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True
        )
        return val_loss

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
