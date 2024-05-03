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

from pytorch_lightning import LightningModule
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from gluonts.torch.model.lightning_util import has_validation_loop

from gluonts.core.component import validated

from .module import MQCNNModel


class MQCNNLightningModule(LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``MQCNNModel`` with PyTorch Lightning.
    This is a thin layer around a (wrapped) ``MQCNNModel`` object,
    that exposes the methods to evaluate training and validation loss.
    Parameters
    ----------
    model_kwargs
        Keyword arguments for the ``MQCNNModel`` object.
    lr
        Learning rate.
    learning_rate_decay_factor
        Learning rate decay factor.
    minimum_learning_rate
        Minimum learning rate.
    weight_decay
        Weight decay regularization parameter.
    patience
        Patience parameter for learning rate scheduler.
    """

    @validated()
    def __init__(
        self,
        model_kwargs: dict,
        lr: float = 1e-3,
        learning_rate_decay_factor: float = 0.9,
        minimum_learning_rate: float = 1e-6,
        weight_decay: float = 1e-8,
        patience: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = MQCNNModel(**model_kwargs)
        self.lr = lr
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.minimum_learning_rate = minimum_learning_rate
        self.weight_decay = weight_decay
        self.patience = patience

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        past_target = batch["past_target"]
        future_target = batch["future_target"]
        past_feat_dynamic = batch["past_feat_dynamic"]
        future_feat_dynamic = batch["future_feat_dynamic"]
        feat_static_real = batch["feat_static_real"]
        feat_static_cat = batch["feat_static_cat"]
        past_observed_values = batch["past_observed_values"]
        future_observed_values = batch["future_observed_values"]
        past_feat_dynamic_cat = batch["past_feat_dynamic_cat"]
        future_feat_dynamic_cat = batch["future_feat_dynamic_cat"]

        loss = self.model.loss(
            past_target=past_target,
            future_target=future_target,
            past_feat_dynamic=past_feat_dynamic,
            future_feat_dynamic=future_feat_dynamic,
            feat_static_real=feat_static_real,
            feat_static_cat=feat_static_cat,
            past_observed_values=past_observed_values,
            future_observed_values=future_observed_values,
            past_feat_dynamic_cat=past_feat_dynamic_cat,
            future_feat_dynamic_cat=future_feat_dynamic_cat,
        )

        # Log every step and epoch, synchronize every epoch
        train_loss = loss.mean()
        self.log(
            "train/loss",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

        return train_loss

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute validation step.
        """
        past_target = batch["past_target"]
        future_target = batch["future_target"]
        past_feat_dynamic = batch["past_feat_dynamic"]
        future_feat_dynamic = batch["future_feat_dynamic"]
        feat_static_real = batch["feat_static_real"]
        feat_static_cat = batch["feat_static_cat"]
        past_observed_values = batch["past_observed_values"]
        future_observed_values = batch["future_observed_values"]
        past_feat_dynamic_cat = batch["past_feat_dynamic_cat"]
        future_feat_dynamic_cat = batch["future_feat_dynamic_cat"]

        loss = self.model.loss(
            past_target=past_target,
            future_target=future_target,
            past_feat_dynamic=past_feat_dynamic,
            future_feat_dynamic=future_feat_dynamic,
            feat_static_real=feat_static_real,
            feat_static_cat=feat_static_cat,
            past_observed_values=past_observed_values,
            future_observed_values=future_observed_values,
            past_feat_dynamic_cat=past_feat_dynamic_cat,
            future_feat_dynamic_cat=future_feat_dynamic_cat,
        )

        # Log and synchronize every epoch
        val_loss = loss.mean()
        self.log(
            "val/loss",
            val_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

        return val_loss

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        monitor = (
            "val/loss" if has_validation_loop(self.trainer) else "train/loss"
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode="min",
                    factor=self.learning_rate_decay_factor,
                    patience=self.patience,
                    min_lr=self.minimum_learning_rate,
                ),
                "monitor": monitor,
            },
        }
