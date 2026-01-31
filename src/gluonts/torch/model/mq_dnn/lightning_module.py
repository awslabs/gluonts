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

from .module import MQDNNModel


class MQDNNLightningModule(pl.LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train an ``MQDNNModel``
    with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``MQDNNModel`` object, that exposes
    the methods to evaluate training and validation loss.

    Parameters
    ----------
    model_kwargs
        Keyword arguments to construct the ``MQDNNModel`` to be trained.
    lr
        Learning rate (default: 1e-3).
    weight_decay
        Weight decay regularization parameter (default: 1e-8).
    patience
        Patience parameter for learning rate scheduler (default: 10).
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
        self.model = MQDNNModel(**model_kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.inputs = self.model.describe_inputs()
        self.example_input_array = self.inputs.zeros()
        self._lazy_layers_materialized = False

    def forward(self, *args, **kwargs):
        """
        Forward pass through the model.
        """
        return self.model(*args, **kwargs)

    def _materialize_lazy_layers(self, batch):
        """
        Materialize lazy layers (LazyConv1d, LazyLinear, lazy RNN) by running
        a dummy forward pass. This ensures all parameters are registered with
        the optimizer before training begins.
        """
        if not self._lazy_layers_materialized:
            with torch.no_grad():
                # Run forward pass to materialize lazy layers
                try:
                    _ = self.model.loss(
                        **select(self.inputs, batch),
                        future_observed_values=batch["future_observed_values"],
                        future_target=batch["future_target"],
                    )
                except Exception:
                    # If forward pass fails, layers might still be materialized
                    pass
            self._lazy_layers_materialized = True

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.

        Parameters
        ----------
        batch
            Training batch.
        batch_idx
            Batch index.

        Returns
        -------
        torch.Tensor
            Training loss.
        """
        # Materialize lazy layers on first batch
        if not self._lazy_layers_materialized:
            self._materialize_lazy_layers(batch)

        # Loss returns shape (batch, prediction_length) - per-timestep loss
        # This matches MXNet's weighted_average over forking dimension
        loss_per_timestep = self.model.loss(
            **select(self.inputs, batch),
            future_observed_values=batch["future_observed_values"],
            future_target=batch["future_target"],
        )
        # MXNet's Loss metric sums the entire tensor, then divides by number of elements
        # sum_metric += tensor.sum(), num_inst += tensor.size
        # result = sum_metric / num_inst = tensor.sum() / tensor.size = tensor.mean()
        # So we just need to take the mean over all dimensions
        train_loss = loss_per_timestep.mean()

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

        Parameters
        ----------
        batch
            Validation batch.
        batch_idx
            Batch index.

        Returns
        -------
        torch.Tensor
            Validation loss.
        """
        # Loss returns shape (batch, prediction_length) - per-timestep loss
        # This matches MXNet's weighted_average over forking dimension
        loss_per_timestep = self.model.loss(
            **select(self.inputs, batch),
            future_observed_values=batch["future_observed_values"],
            future_target=batch["future_target"],
        )
        # MXNet's Loss metric sums the entire tensor, then divides by number of elements
        # sum_metric += tensor.sum(), num_inst += tensor.size
        # result = sum_metric / num_inst = tensor.sum() / tensor.size = tensor.mean()
        # So we just need to take the mean over all dimensions
        val_loss = loss_per_timestep.mean()

        self.log(
            "val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True
        )

        return val_loss

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        IMPORTANT: Materializes lazy layers before creating optimizer to ensure
        all parameters are registered.

        Returns
        -------
        dict
            Dictionary with optimizer and lr_scheduler configuration.
        """
        # Materialize lazy layers if not already done
        if not self._lazy_layers_materialized:
            # Create example batch to materialize layers
            # Note: example_input_array already has batch_size=1, so we don't add another dimension
            example_batch = dict(self.example_input_array)

            # Add required training fields with proper shapes
            batch_size = 1
            example_batch["future_target"] = torch.zeros(
                batch_size, self.model.num_forking, self.model.prediction_length
            )
            example_batch["future_observed_values"] = torch.ones(
                batch_size, self.model.num_forking, self.model.prediction_length
            )

            # Materialize with dummy forward pass
            with torch.no_grad():
                try:
                    _ = self.model.loss(
                        **select(self.inputs, example_batch),
                        future_observed_values=example_batch["future_observed_values"],
                        future_target=example_batch["future_target"],
                    )
                    self._lazy_layers_materialized = True
                except Exception as e:
                    # Log warning but continue - layers might still be partially materialized
                    import warnings
                    warnings.warn(f"Failed to materialize lazy layers in configure_optimizers: {e}")

        # Now create optimizer with all parameters (including materialized lazy ones)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Check if trainer exists before calling has_validation_loop
        try:
            monitor = "val_loss" if has_validation_loop(self.trainer) else "train_loss"
        except RuntimeError:
            # If trainer is not attached, default to train_loss
            monitor = "train_loss"

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
