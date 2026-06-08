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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from gluonts.core.component import validated
from gluonts.itertools import select
from gluonts.torch.model.lightning_util import has_validation_loop

from .module import SMTModel


class SMTLightningModule(pl.LightningModule):
    """
    A ``pl.LightningModule`` to train an ``SMTModel`` with PyTorch Lightning.

    The training and validation losses are the total SMT objective
    (predictive-state NLL + dynamics MSE + uniformity); the individual
    components are logged for monitoring.

    Parameters
    ----------
    model_kwargs
        Keyword arguments to construct the ``SMTModel`` to be trained.
    lr
        Learning rate.
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
        weight_decay: float = 1e-8,
        patience: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = SMTModel(**model_kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.inputs = self.model.describe_inputs()
        self.example_input_array = self.inputs.zeros()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _step(self, batch, prefix: str):
        losses = self.model.loss(
            **select(self.inputs, batch),
            future_observed_values=batch["future_observed_values"],
            future_target=batch["future_target"],
        )
        for name, value in losses.items():
            key = f"{prefix}_loss" if name == "loss" else f"{prefix}_{name}"
            self.log(
                key,
                value,
                on_epoch=True,
                on_step=False,
                prog_bar=name == "loss",
            )
        return losses["loss"]

    def training_step(self, batch, batch_idx: int):  # type: ignore
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        monitor = (
            "val_loss" if has_validation_loop(self.trainer) else "train_loss"
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
                "monitor": monitor,
            },
        }
