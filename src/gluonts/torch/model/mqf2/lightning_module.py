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

from typing import Dict

import torch

from gluonts.core.component import validated
from gluonts.torch.modules.loss import DistributionLoss, EnergyScore
from gluonts.torch.model.deepar.lightning_module import DeepARLightningModule
from . import MQF2MultiHorizonModel


class MQF2MultiHorizonLightningModule(DeepARLightningModule):
    r"""
    LightningModule class for the model MQF2 proposed in the paper
    ``Multivariate Quantile Function Forecaster``
    by Kan, Aubet, Januschowski, Park, Benidis, Ruthotto, Gasthaus

    This is the multi-horizon (multivariate in time step) variant of MQF2

    This class is based on
    gluonts.torch.model.deepar.lightning_module.DeepARLightningModule

    Parameters
    ----------
    model
        An MQF2MultiHorizonModel instance
    loss
        Distribution loss
    lr
        Learning rate
    weight_decay
        Weight decay during training
    patience
        Patience parameter for learning rate scheduler, default: ``10``.
    """

    @validated()
    def __init__(
        self,
        model: MQF2MultiHorizonModel,
        loss: DistributionLoss = EnergyScore(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
    ) -> None:
        super().__init__(
            model=model,
            loss=loss,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
        )

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Function to compute loss.

        Given time series, unroll the RNN over the context window
        and pass the hidden states to the forecaster
        then the loss with respect to the prediction is computed

        Parameters
        ----------
        batch
            Dictionary containing the (past and future) features
            and target values in a batch

        Returns
        -------
        loss
            Mean of the loss values
        """

        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]
        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        past_observed_values = batch["past_observed_values"]

        picnn = self.model.picnn

        hidden_state, scale = self.model.unroll_lagged_rnn(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )

        distr = self.model.output_distribution(picnn, hidden_state, scale)

        context_target = past_target[:, -self.model.context_length + 1 :]
        target = torch.cat(
            (context_target, future_target),
            dim=1,
        )

        loss_values = self.loss(distr, target)

        return loss_values.mean()
