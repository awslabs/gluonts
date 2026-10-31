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

from .module import CrossformerModel


class CrossformerLightningModule(pl.LightningModule):
    @validated()
    def __init__(
        self,
        model_kwargs: dict,
        num_parallel_samples: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = CrossformerModel(**model_kwargs)
        self.num_parallel_samples = num_parallel_samples
        self.lr = lr
        self.weight_decay = weight_decay
        self.inputs = self.model.describe_inputs()

    def forward(self, *args, **kwargs):
        distr_args, loc, scale = self.model(*args, **kwargs)
        if hasattr(self.model.distr_output, "distribution"):
            distr = self.model.distr_output.distribution(
                distr_args, loc, scale
            )
            samples = distr.sample((self.num_parallel_samples,))
            return samples.transpose(0, 1)
        return distr_args, loc, scale

    def training_step(self, batch, batch_idx: int):  # type: ignore
        train_loss = self.model.loss(
            **select(self.inputs, batch),
            future_target=batch["future_target"],
            future_observed_values=batch["future_observed_values"],
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
        val_loss = self.model.loss(
            **select(self.inputs, batch),
            future_target=batch["future_target"],
            future_observed_values=batch["future_observed_values"],
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
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
