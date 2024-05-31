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

from typing import List, Tuple

import torch
from torch import nn

from gluonts.core.component import validated
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.model import Input, InputSpec
from gluonts.torch.distributions import Output
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.model.simple_feedforward import make_linear_layer
from gluonts.torch.util import weighted_average


class ResBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        dropout_rate: float,
        layer_norm: bool,
    ):
        super().__init__()

        self.fc = nn.Sequential(
            make_linear_layer(dim_in, dim_hidden),
            nn.ReLU(),
            make_linear_layer(dim_hidden, dim_out),
            nn.Dropout(p=dropout_rate),
        )
        self.skip = make_linear_layer(dim_in, dim_out)
        if layer_norm:
            self.ln = nn.LayerNorm(dim_out)
        self.layer_norm = layer_norm

    def forward(self, x):
        if self.layer_norm:
            return self.ln(self.fc(x) + self.skip(x))
        return self.fc(x) + self.skip(x)


class FeatureProjection(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float,
        layer_norm: bool,
    ):
        super().__init__()

        self.proj = ResBlock(
            input_dim,
            hidden_dim,
            output_dim,
            dropout_rate,
            layer_norm,
        )

    def forward(self, x):
        return self.proj(x)


class DenseEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        dropout_rate: float,
        layer_norm: bool,
    ):
        super().__init__()

        layers = []
        layers.append(
            ResBlock(
                input_dim, hidden_dim, hidden_dim, dropout_rate, layer_norm
            )
        )
        for i in range(num_layers - 1):
            layers.append(
                ResBlock(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    dropout_rate,
                    layer_norm,
                )
            )
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class DenseDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float,
        layer_norm: bool,
    ):
        super().__init__()

        layers = []
        for i in range(num_layers - 1):
            layers.append(
                ResBlock(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    dropout_rate,
                    layer_norm,
                )
            )
        layers.append(
            ResBlock(
                hidden_dim,
                hidden_dim,
                output_dim,
                dropout_rate,
                layer_norm,
            )
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class TemporalDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float,
        layer_norm: bool,
    ):
        super().__init__()

        self.temporal_decoder = ResBlock(
            input_dim,
            hidden_dim,
            output_dim,
            dropout_rate,
            layer_norm,
        )

    def forward(self, x):
        return self.temporal_decoder(x)


class TiDEModel(nn.Module):
    """

    Parameters
    ----------
    context_length
        Number of time steps prior to prediction time that the model
        takes as inputs.
    prediction_length
        Length of the prediction horizon.
    num_feat_dynamic_proj
        Output size of feature projection layer.
    num_feat_dynamic_real
        Number of dynamic real features in the data.
    num_feat_static_real
        Number of static real features in the data.
    num_feat_static_cat
        Number of static categorical features in the data.
    cardinality
        Number of values of each categorical feature.
        This must be set if ``num_feat_static_cat > 0``.
    embedding_dimension
        Dimension of the embeddings for categorical features.
    feat_proj_hidden_dim
        Size of the feature projection layer.
    encoder_hidden_dim
        Size of the dense encoder layer.
    decoder_hidden_dim
        Size of the dense decoder layer.
    temporal_hidden_dim
        Size of the temporal decoder layer.
    distr_hidden_dim
        Size of the distribution projection layer.
    decoder_output_dim
        Output size of dense decoder.
    dropout_rate
        Dropout regularization parameter.
    num_layers_encoder
        Number of layers in dense encoder.
    num_layers_decoder
        Number of layers in dense decoder.
    layer_norm
        Enable layer normalization or not.
    distr_output
        Distribution to use to evaluate observations and sample predictions.
    scaling
        Which scaling method to use to scale the target values.


    """

    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int,
        num_feat_dynamic_proj: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        cardinality: List[int],
        embedding_dimension: List[int],
        feat_proj_hidden_dim: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        temporal_hidden_dim: int,
        distr_hidden_dim: int,
        decoder_output_dim: int,
        dropout_rate: float,
        num_layers_encoder: int,
        num_layers_decoder: int,
        layer_norm: bool,
        distr_output: Output,
        scaling: str,
    ) -> None:
        super().__init__()

        assert context_length > 0
        assert prediction_length > 0
        assert num_feat_dynamic_real > 0
        assert num_feat_static_real > 0
        assert num_feat_static_cat > 0
        assert len(cardinality) == num_feat_static_cat
        assert len(embedding_dimension) == num_feat_static_cat

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_dynamic_proj = num_feat_dynamic_proj
        self.num_feat_static_real = num_feat_static_real
        self.num_feat_static_cat = num_feat_static_cat
        self.embedding_dimension = embedding_dimension
        self.feat_proj_hidden_dim = feat_proj_hidden_dim
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.distr_hidden_dim = distr_hidden_dim
        self.decoder_output_dim = decoder_output_dim
        self.dropout_rate = dropout_rate

        self.proj_flatten_dim = (
            context_length + prediction_length
        ) * num_feat_dynamic_proj
        encoder_input_dim = (
            context_length
            + num_feat_static_real
            + sum(self.embedding_dimension)
            + self.proj_flatten_dim
        )
        self.temporal_decoder_input_dim = (
            decoder_output_dim + num_feat_dynamic_proj
        )

        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )

        self.feat_proj = FeatureProjection(
            num_feat_dynamic_real,
            feat_proj_hidden_dim,
            num_feat_dynamic_proj,
            dropout_rate,
            layer_norm,
        )
        self.dense_encoder = DenseEncoder(
            num_layers_encoder,
            encoder_input_dim,
            encoder_hidden_dim,
            dropout_rate,
            layer_norm,
        )
        self.dense_decoder = DenseDecoder(
            num_layers_encoder,
            decoder_hidden_dim,
            prediction_length * decoder_output_dim,
            dropout_rate,
            layer_norm,
        )
        self.temporal_decoder = TemporalDecoder(
            self.temporal_decoder_input_dim,
            temporal_hidden_dim,
            distr_hidden_dim,
            dropout_rate,
            layer_norm,
        )
        self.loopback_skip = make_linear_layer(
            self.context_length, self.prediction_length * distr_hidden_dim
        )

        self.distr_output = distr_output
        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        self.args_proj = self.distr_output.get_args_proj(self.distr_hidden_dim)

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
                "past_time_feat": Input(
                    shape=(
                        batch_size,
                        self.context_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
                "feat_static_real": Input(
                    shape=(batch_size, self.num_feat_static_real),
                    dtype=torch.float,
                ),
                "feat_static_cat": Input(
                    shape=(batch_size, self.num_feat_static_cat),
                    dtype=torch.long,
                ),
                "future_time_feat": Input(
                    shape=(
                        batch_size,
                        self.prediction_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
            },
            torch.zeros,
        )

    def forward(
        self,
        feat_static_real: torch.Tensor,
        feat_static_cat: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        past_target_scaled, loc, scale = self.scaler(
            past_target, past_observed_values
        )

        embedded_cat = self.embedder(feat_static_cat)
        time_feat = torch.cat((past_time_feat, future_time_feat), dim=-2)
        proj = self.feat_proj(time_feat)
        proj_future = proj[..., -self.prediction_length :, :]
        proj_flatten = proj.view(-1, self.proj_flatten_dim)
        encoder_input = torch.cat(
            (past_target_scaled, feat_static_real, embedded_cat, proj_flatten),
            dim=-1,
        )
        encoder_output = self.dense_encoder(encoder_input)
        decoder_output = self.dense_decoder(encoder_output)

        temporal_decoder_input = torch.cat(
            (
                decoder_output.view(
                    -1, self.prediction_length, self.decoder_output_dim
                ),
                proj_future,
            ),
            dim=-1,
        )
        out = self.temporal_decoder(temporal_decoder_input)
        out = (
            self.loopback_skip(past_target_scaled).view(
                -1, self.prediction_length, self.distr_hidden_dim
            )
            + out
        )

        distr_args = self.args_proj(out)
        return distr_args, loc, scale

    def loss(
        self,
        feat_static_real: torch.Tensor,
        feat_static_cat: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
    ):
        distr_args, loc, scale = self(
            feat_static_real=feat_static_real,
            feat_static_cat=feat_static_cat,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
        )
        loss = self.distr_output.loss(
            target=future_target, distr_args=distr_args, loc=loc, scale=scale
        )
        return weighted_average(loss, weights=future_observed_values, dim=-1)
