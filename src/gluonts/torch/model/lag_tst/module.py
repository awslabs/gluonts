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

from typing import Tuple, Optional, List

import torch
from torch import nn

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.util import unsqueeze_expand, lagged_sequence_values
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.model.simple_feedforward import make_linear_layer
from gluonts.torch.model.patch_tst import SinusoidalPositionalEmbedding


class LagTSTModel(nn.Module):
    """
    Module implementing the LagTST model for forecasting.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    context_length
        Number of time steps prior to prediction time that the model.
    distr_output
        Distribution to use to evaluate observations and sample predictions.
        Default: ``StudentTOutput()``.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        freq: str,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        num_encoder_layers: int,
        scaling: str,
        lags_seq: Optional[List[int]] = None,
        distr_output=StudentTOutput(),
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.lags_seq = lags_seq or get_lags_for_frequency(
            freq_str=freq, num_default_lags=1
        )
        self.d_model = d_model
        self.distr_output = distr_output

        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        # project from number of lags + 2 features (loc and scale) to d_model
        self.patch_proj = make_linear_layer(len(self.lags_seq) + 2, d_model)

        self.positional_encoding = SinusoidalPositionalEmbedding(
            self.context_length, d_model
        )

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

        self.flatten = nn.Linear(
            d_model * self.context_length, prediction_length * d_model
        )

        self.args_proj = self.distr_output.get_args_proj(d_model)

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self._past_length), dtype=torch.float
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self._past_length), dtype=torch.float
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

        lags = lagged_sequence_values(
            self.lags_seq,
            past_target_scaled[:, : -self.context_length, ...],
            past_target_scaled[:, -self.context_length :, ...],
            dim=-1,
        )

        # add loc and scale to past_target_patches as additional features
        log_abs_loc = loc.abs().log1p()
        log_scale = scale.log()
        expanded_static_feat = unsqueeze_expand(
            torch.cat([log_abs_loc, log_scale], dim=-1),
            dim=1,
            size=lags.shape[1],
        )
        inputs = torch.cat((lags, expanded_static_feat), dim=-1)

        # project patches
        enc_in = self.patch_proj(inputs)
        embed_pos = self.positional_encoding(enc_in.size())

        # transformer encoder with positional encoding
        enc_out = self.encoder(enc_in + embed_pos)

        # flatten and project to prediction length * d_model
        flatten_out = self.flatten(enc_out.flatten(start_dim=1))

        # project to distribution arguments
        distr_args = self.args_proj(
            flatten_out.reshape(-1, self.prediction_length, self.d_model)
        )
        return distr_args, loc, scale
