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
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood

from .module import DLinearModel


class DLinearLightningModule(pl.LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``DLinearModel`` with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``DLinearModel`` object,
    that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model_kwargs
        Keyword arguments to construct the ``DLinearModel`` to be trained.
    loss
        Loss function to be used for training.
    lr
        Learning rate.
    weight_decay
        Weight decay regularization parameter.
    """

    @validated()
    def __init__(
        self,
        model_kwargs: dict,
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = DLinearModel(**model_kwargs)
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def _compute_loss(self, batch):
        context = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        target = batch["future_target"]
        observed_target = batch["future_observed_values"]

        assert context.shape[-1] == self.model.context_length
        assert target.shape[-1] == self.model.prediction_length

        distr_args, loc, scale = self.model(
            context, past_observed_values=past_observed_values
        )
        distr = self.model.distr_output.distribution(distr_args, loc, scale)

        return (
            self.loss(distr, target) * observed_target
        ).sum() / observed_target.sum().clamp_min(1.0)

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
