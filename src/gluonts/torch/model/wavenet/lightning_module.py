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

import lightning.pytorch as pl
import torch

from gluonts.core.component import validated

from .module import WaveNet


class WaveNetLightningModule(pl.LightningModule):
    """
    LightningModule wrapper over WaveNet.

    Parameters
    ----------
    model_kwargs
        Keyword arguments to pass to WaveNet.
    lr, optional
        Learning rate, by default 1e-3
    weight_decay, optional
        Weight decay, by default 1e-8
    """

    @validated()
    def __init__(
        self,
        model_kwargs: dict,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = WaveNet(**model_kwargs)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        past_time_feat = batch["past_time_feat"]
        future_target = batch["future_target"]
        future_observed_values = batch["future_observed_values"]
        future_time_feat = batch["future_time_feat"]
        scale = batch["scale"]

        train_loss = self.model.loss(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_target=future_target,
            future_observed_values=future_observed_values,
            future_time_feat=future_time_feat,
            scale=scale,
        ).mean()

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
        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        past_time_feat = batch["past_time_feat"]
        future_target = batch["future_target"]
        future_observed_values = batch["future_observed_values"]
        future_time_feat = batch["future_time_feat"]
        scale = batch["scale"]

        val_loss = self.model.loss(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_target=future_target,
            future_observed_values=future_observed_values,
            future_time_feat=future_time_feat,
            scale=scale,
        ).mean()

        self.log(
            "val_loss",
            val_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        return val_loss

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
