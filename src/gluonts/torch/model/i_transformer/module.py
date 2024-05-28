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

from typing import Optional, Tuple

import torch
from torch import nn

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.util import weighted_average


class ITransformerModel(nn.Module):
    """
    Module implementing the iTransformer model for multivariate forecasting as
    described in https://arxiv.org/abs/2310.06625 extended to be probabilistic.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    context_length
        Number of time steps prior to prediction time that the model.
    d_model
        Transformer latent dimension.
    nhead
        Number of attention heads which must be divisible with d_model.
    dim_feedforward
        Dimension of the transformer's feedforward network model.
    dropout
        Dropout rate for the transformer.
    activation
        Activation function for the transformer.
    norm_first
        Whether to normalize the input before the transformer.
    num_encoder_layers
        Number of transformer encoder layers.
    scaling
        Whether to scale the input using mean or std or None.
    distr_output
        Distribution to use to evaluate observations and sample predictions.
        Default: ``StudentTOutput()``.
    nonnegative_pred_samples
        Should final prediction samples be non-negative? If yes, an activation
        function is applied to ensure non-negative. Observe that this is applied
        only to the final samples and this is not applied during training.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        num_encoder_layers: int,
        scaling: Optional[str],
        distr_output=StudentTOutput(),
        nonnegative_pred_samples: bool = False,
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0

        self.prediction_length = prediction_length
        self.context_length = context_length

        self.d_model = d_model
        self.nhead = nhead
        self.distr_output = distr_output

        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True, dim=1)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)
        self.nonnegative_pred_samples = nonnegative_pred_samples

        # project each variate plus mean and std to d_model dimension
        self.emebdding = nn.Linear(context_length + 2, d_model)

        # transformer encoder
        layer_norm_eps: float = 1e-5
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=norm_first,
        )
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        # project each variate to prediction length number of latent variables
        self.projection = nn.Linear(
            d_model, prediction_length * d_model // nhead
        )

        # project each prediction length latent to distribution parameters
        self.args_proj = self.distr_output.get_args_proj(d_model // nhead)

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self.context_length, -1),
                    dtype=torch.float,
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.context_length, -1),
                    dtype=torch.float,
                ),
            },
            torch.zeros,
        )

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        # scale the input
        past_target_scaled, loc, scale = self.scaler(
            past_target, past_observed_values
        )
        log_abs_loc = loc.abs().log1p()
        log_scale = scale.log()

        # Transpose to time last
        past_target_scaled = past_target_scaled.transpose(1, 2)
        log_abs_loc = log_abs_loc.transpose(1, 2)
        log_scale = log_scale.transpose(1, 2)

        # concatenate past target with log_abs_loc and log_scale
        expanded_target_scaled = torch.cat(
            [past_target_scaled, log_abs_loc, log_scale], dim=-1
        )

        # project to d_model
        enc_in = self.emebdding(expanded_target_scaled)

        # transformer encoder with positional encoding
        enc_out = self.encoder(enc_in)

        # project to prediction length * d_model // nhead
        projection_out = self.projection(enc_out).reshape(
            -1,
            past_target.shape[2],
            self.prediction_length,
            self.d_model // self.nhead,
        )

        # transpose to prediction length first
        projection_out_transpose = projection_out.transpose(1, 2)

        # project to distribution arguments
        distr_args = self.args_proj(projection_out_transpose)
        return distr_args, loc, scale

    def loss(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> torch.Tensor:
        distr_args, loc, scale = self(
            past_target=past_target, past_observed_values=past_observed_values
        )
        loss = self.distr_output.loss(
            target=future_target, distr_args=distr_args, loc=loc, scale=scale
        )
        return weighted_average(loss, weights=future_observed_values, dim=-1)
