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
from torch.nn import functional as F

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.scaler import MeanScaler, NOPScaler, StdScaler
from gluonts.torch.util import weighted_average


class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input):
        batch_size, channels, _ = input.shape

        # set FFN
        combined_mean = F.gelu(self.gen1(input))
        combined_mean = self.gen2(combined_mean)

        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(
                combined_mean * weight, dim=1, keepdim=True
            ).repeat(1, channels, 1)

        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        output = combined_mean_cat

        return output


class EncoderLayer(nn.Module):
    def __init__(
        self,
        star,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.star = star
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x = self.star(x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class SofTSModel(nn.Module):
    """
    Module implementing the SOFTS model for multivariate forecasting as
    described in https://arxiv.org/pdf/2404.14197 extended to be probabilistic.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    context_length
        Number of time steps prior to prediction time that the model.
    d_model
        Size of latent in the encoder.
    d_core
        Dimension of the global core representation.
    dim_feedforward
        Dimension of the encoder's feedforward network model.
    dim_projections
        Dimension of the projection layer.
    dropout
        Dropout rate for the encoder.
    activation
        Activation function for the encoder.
    num_encoder_layers
        Number of STAR layers in the encoder.
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
        d_core: int,
        dim_feedforward: int,
        dim_projections: int,
        dropout: float,
        activation: str,
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

        self.dim_projections = dim_projections
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

        # star encoder
        self.encoder = nn.Sequential(
            *[
                EncoderLayer(
                    STAR(d_series=d_model, d_core=d_core),
                    d_model=d_model,
                    d_ff=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # project each variate to prediction length number of latent variables
        self.projection = nn.Linear(
            d_model, prediction_length * dim_projections
        )

        # project each prediction length latent to distribution parameters
        self.args_proj = self.distr_output.get_args_proj(dim_projections)

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
        log_abs_loc = loc.sign() * loc.abs().log1p()
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

        # project to prediction length * dim_projections
        projection_out = self.projection(enc_out).reshape(
            -1,
            past_target.shape[2],
            self.prediction_length,
            self.dim_projections,
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
