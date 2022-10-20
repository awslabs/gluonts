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

from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.functional import softplus

from gluonts.itertools import prod
from gluonts.torch.modules.scaler import MeanScaler


class MQDNNModel(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        num_output_quantiles: int,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        encoder: nn.Module,
        encoder_output_length: int,
        decoder_latent_length: int,
        global_hidden_sizes: List[int],
        local_hidden_sizes: List[int],
        global_activation: nn.Module = nn.ReLU(),
        local_activation: nn.Module = nn.ReLU(),
        scaling: nn.Module = MeanScaler(dim=1, keepdim=True),
        is_iqf: bool = False,
    ) -> None:
        super().__init__()
        self.prediction_length = prediction_length
        self.num_output_quantiles = num_output_quantiles
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_real = num_feat_static_real
        self.encoder = encoder
        self.encoder_output_length = encoder_output_length
        self.decoder_latent_length = decoder_latent_length
        self.global_hidden_sizes = global_hidden_sizes
        self.local_hidden_sizes = local_hidden_sizes
        self.global_activation = global_activation
        self.local_activation = local_activation

        self.global_decoder = self._make_global_decoder()
        self.local_decoder = self._make_local_decoder()

        self.scaling = scaling
        self.is_iqf = is_iqf

    def _make_global_decoder(self):
        input_length = self.encoder_output_length + (
            self.prediction_length * self.num_feat_dynamic_real
        )
        dimensions = [input_length] + self.global_hidden_sizes

        modules = []
        for in_size, out_size in zip(dimensions[:-1], dimensions[1:]):
            modules += [nn.Linear(in_size, out_size), self.global_activation]

        output_shape = (
            self.prediction_length + 1,
            self.decoder_latent_length,
        )

        modules += [
            nn.Linear(dimensions[-1], prod(output_shape)),
            nn.Unflatten(dim=1, unflattened_size=output_shape),
        ]

        return nn.Sequential(*modules)

    def _make_local_decoder(self):
        input_length = 2 * self.decoder_latent_length + (
            self.num_feat_dynamic_real
        )
        dimensions = [input_length] + self.local_hidden_sizes

        modules = []
        for in_size, out_size in zip(dimensions[:-1], dimensions[1:]):
            modules += [nn.Linear(in_size, out_size), self.local_activation]

        modules.append(
            nn.Linear(dimensions[-1], self.num_output_quantiles),
        )
        modules.append(Transpose(1, 2))

        return nn.Sequential(*modules)

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: Optional[torch.Tensor] = None,
        past_feat_dynamic: Optional[torch.Tensor] = None,
        future_feat_dynamic: Optional[torch.Tensor] = None,
        feat_static_real: Optional[torch.Tensor] = None,
        feat_static_cat: Optional[torch.Tensor] = None,
    ):
        assert (past_feat_dynamic is None) == (future_feat_dynamic is None)

        if past_observed_values is None:
            past_observed_values = torch.ones_like(past_target)
        scaled_past_target, scale = self.scaling(
            past_target, past_observed_values
        )

        # TODO add static categorical features as well

        past_time_length = past_observed_values.shape[1]

        encoder_inputs = [
            scaled_past_target.unsqueeze(-1),
            past_observed_values.unsqueeze(-1),
            torch.log(scale).unsqueeze(-1).repeat(1, past_time_length, 1),
        ]
        if past_feat_dynamic is not None:
            encoder_inputs.append(past_feat_dynamic)
        if feat_static_real is not None:
            encoder_inputs.append(
                feat_static_real.unsqueeze(1).repeat(1, past_time_length, 1)
            )

        encoder_output = self.encoder(torch.concat(encoder_inputs, dim=-1))

        global_decoder_inputs = [encoder_output[:, -1]]
        if future_feat_dynamic is not None:
            global_decoder_inputs.append(
                future_feat_dynamic.flatten(start_dim=1)
            )

        global_decoder_output = self.global_decoder(
            torch.concat(global_decoder_inputs, dim=1)
        )

        local_decoder_inputs = [
            global_decoder_output[:, :-1],
            global_decoder_output[:, -1:].expand(
                -1, self.prediction_length, -1
            ),
        ]
        if future_feat_dynamic is not None:
            local_decoder_inputs.append(future_feat_dynamic)

        local_decoder_output = self.local_decoder(
            torch.concat(local_decoder_inputs, dim=2)
        )

        if self.is_iqf:
            quantiles = torch.concat(
                (
                    local_decoder_output[:, 0:1, :],
                    softplus(local_decoder_output[:, 1:, :]),
                ),
                dim=1,
            ).cumsum(dim=1) * scale.unsqueeze(-1)
        else:
            quantiles = local_decoder_output * scale.unsqueeze(-1)

        return quantiles


class Transpose(torch.nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Select(torch.nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        return x[self.idx]


class MQCNNModel(MQDNNModel):
    def __init__(
        self,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        encoder_channels: List[int] = [30, 30, 30],
        encoder_dilations: List[int] = [1, 3, 9],
        encoder_kernel_sizes: List[int] = [7, 3, 3],
        **kwargs,
    ) -> None:
        assert len(encoder_channels) > 0
        assert len(encoder_dilations) == len(encoder_channels)
        assert len(encoder_kernel_sizes) == len(encoder_channels)

        target_features = 3
        input_channels = (
            num_feat_dynamic_real + num_feat_static_real + target_features
        )
        channels = [input_channels] + encoder_channels
        in_out_channels = zip(channels[:-1], channels[1:])

        conv_layers = [
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            for (in_channels, out_channels), dilation, kernel_size in zip(
                in_out_channels, encoder_dilations, encoder_kernel_sizes
            )
        ]

        encoder_layers = [Transpose(1, 2)]
        for conv_layer in conv_layers[:-1]:
            encoder_layers.append(conv_layer)
            encoder_layers.append(torch.nn.ReLU())
        encoder_layers.append(conv_layers[-1])
        encoder_layers.append(Transpose(1, 2))

        encoder = torch.nn.Sequential(*encoder_layers)

        super().__init__(
            num_feat_dynamic_real=num_feat_dynamic_real,
            num_feat_static_real=num_feat_static_real,
            encoder=encoder,
            encoder_output_length=channels[-1],
            **kwargs,
        )


class MQRNNModel(MQDNNModel):
    def __init__(
        self,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        num_layers: int = 1,
        encoder_hidden_size: int = 50,
        **kwargs,
    ) -> None:
        target_features = 3
        input_size = (
            num_feat_dynamic_real + num_feat_static_real + target_features
        )
        encoder = torch.nn.Sequential(
            torch.nn.LSTM(
                input_size=input_size,
                hidden_size=encoder_hidden_size,
                num_layers=num_layers,
                batch_first=True,
            ),
            Select(0),
        )

        super().__init__(
            num_feat_dynamic_real=num_feat_dynamic_real,
            num_feat_static_real=num_feat_static_real,
            encoder=encoder,
            encoder_output_length=encoder_hidden_size,
            **kwargs,
        )
