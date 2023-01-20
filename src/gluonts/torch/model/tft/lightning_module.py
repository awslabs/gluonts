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

import pytorch_lightning as pl
import torch
from gluonts.core.component import validated
from gluonts.itertools import select
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .module import TemporalFusionTransformerModel


class TemporalFusionTransformerLightningModule(pl.LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``TemporalFusionTransformerModel`` with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``TemporalFusionTransformerModel``
    object, that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model
        ``TemporalFusionTransformerModel`` to be trained.
    lr
        Learning rate, default: ``1e-3``.
    weight_decay
        Weight decay regularization parameter, default: ``1e-8``.
    patience
        Patience parameter for learning rate scheduler, default: ``10``.
    """

    @validated()
    def __init__(
        self,
        model: TemporalFusionTransformerModel,
        lr: float = 1e-3,
        patience: int = 10,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.patience = patience
        self.weight_decay = weight_decay
        self.example_input_array = tuple(
            [
                torch.zeros(shape, dtype=self.model.input_types()[name])
                for (name, shape) in self.model.input_shapes().items()
            ]
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        train_loss = self.model.loss(
            **select(self.model.input_shapes(), batch, ignore_missing=True),
            future_observed_values=batch["future_observed_values"],
            future_target=batch["future_target"],
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
        val_loss = self.model.loss(
            **select(self.model.input_shapes(), batch, ignore_missing=True),
            future_observed_values=batch["future_observed_values"],
            future_target=batch["future_target"],
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
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode="min",
                    factor=0.5,
                    patience=self.patience,
                ),
                "monitor": "train_loss",
            },
        }
