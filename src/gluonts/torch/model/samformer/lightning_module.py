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
from gluonts.itertools import select

from .module import SamFormerModel
from .sam import SAM


class SamFormerLightningModule(pl.LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a ``SamFormerModel`` with
    PyTorch Lightning.

    This is a thin layer around a (wrapped) ``SamFormerModel`` object,
    that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model_kwargs
        Keyword arguments to construct the ``SamFormerModel`` to be trained.
    num_parallel_samples:
        Number of evaluation samples per time series to sample during inference.
    lr
        Learning rate.
    weight_decay
        Weight decay regularization parameter.
    rho
        Rho parameter for SAM optimizer.
    sam
        Whether to use SAM optimizer.
    """

    @validated()
    def __init__(
        self,
        model_kwargs: dict,
        num_parallel_samples: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        rho: float = 0.5,
        sam: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SamFormerModel(**model_kwargs)
        self.num_parallel_samples = num_parallel_samples
        self.lr = lr
        self.weight_decay = weight_decay
        self.rho = rho
        self.sam = sam

        self.automatic_optimization = False

        self.inputs = self.model.describe_inputs()

    def forward(self, *args, **kwargs):
        distr_args, loc, scale = self.model.forward(*args, **kwargs)
        distr = self.model.distr_output.distribution(distr_args, loc, scale)

        samples = distr.sample((self.num_parallel_samples,))
        if self.model.nonnegative_pred_samples:
            samples = torch.relu(samples)
        return samples.transpose(0, 1)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        opt = self.optimizers()

        train_loss = self.model.loss(
            **select(self.inputs, batch),
            future_target=batch["future_target"],
            future_observed_values=batch["future_observed_values"],
        ).mean()

        if self.sam:
            # Ascent Step
            self.manual_backward(train_loss)
            opt.first_step(zero_grad=True)

            # Descent Step
            train_loss_2 = self.model.loss(
                **select(self.inputs, batch),
                future_target=batch["future_target"],
                future_observed_values=batch["future_observed_values"],
            ).mean()
            self.manual_backward(train_loss_2)
            opt.second_step(zero_grad=True)
        else:
            opt.zero_grad()
            self.manual_backward(train_loss)
            opt.step()

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
            **select(self.inputs, batch),
            future_target=batch["future_target"],
            future_observed_values=batch["future_observed_values"],
        ).mean()

        self.log(
            "val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True
        )
        return val_loss

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        if self.sam:
            return SAM(
                self.model.parameters(),
                base_optimizer=torch.optim.Adam,
                lr=self.lr,
                rho=self.rho,
                weight_decay=self.weight_decay,
            )
        else:
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
